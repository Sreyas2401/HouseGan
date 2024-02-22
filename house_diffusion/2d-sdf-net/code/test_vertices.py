import cv2
import numpy as np

# Read the image
img_path = '../shapes/raw_images/quarter_circle.png'
image = cv2.imread(img_path)
out_path = '../shapes/raw/'
width, height, channels = image.shape

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to obtain binary image
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pts = []
# Iterate through contours
for contour in contours:
    # Approximate polygonal curve
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Optional: Filter out small contours based on area
    if cv2.contourArea(approx) < 100:
        continue
    
    # Get coordinates of the polygonal curve
    pts = np.squeeze(approx).tolist()
    print("Coordinates of the polygon:", pts)
    
    # Draw contour on original image for visualization
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Show the result
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

f = open(f'{out_path}/quarter_circle', 'w')
for point in pts:
    # Normalize (x,y) to (0,1)
    x = (np.double(point[0]) / width) 
    y = (np.double(point[1]) / height) 
    f.write(f'{x} {y}\n')
f.close()
