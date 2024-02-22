import cv2
import numpy as np
import os

def find_rooms(img, noise_removal_threshold=25, corners_threshold=0.1,
               room_closing_max_length=100, gap_in_wall_threshold=500):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    img[img < 128] = 0
    img[img >= 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = cv2.bitwise_and(img, mask)

    cv2.imshow('mask1', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imshow('image1', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img ,2,3,0.04)
    dst = cv2.dilate(dst,None)
    corners = dst > corners_threshold * dst.max()

    # Mark the outside of the house as black
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0

    cv2.imshow('image2', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        component = labels == label
        if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    return rooms, img



#Read gray image
img = cv2.imread("./output_Inverted/1.png", 0)
rooms, colored_house = find_rooms(img.copy())
cv2.imwrite("./output_Color/1_1.png", colored_house)

output_folder = './output_Seperated/1_room_imgs'
os.makedirs(output_folder, exist_ok=True)

# Iterate over each room and save it
for i, room in enumerate(rooms):
    # Create a mask for the current room
    mask = np.zeros_like(room, dtype=np.uint8)
    mask[room] = 255

    # Crop the original image based on the room's mask
    cropped_room = cv2.bitwise_and(img, img, mask=mask)

    # Save the room as an individual image in the output folder
    room_filename = os.path.join(output_folder, f'room_{i}.png')
    cv2.imwrite(room_filename, cropped_room)
