import cv2
import numpy as np

def convert_colors(input_image_path, output_image_path):
    # Read the image
    image = cv2.imread(input_image_path)

    # Define the dark color range (almost black)
    lower_dark = np.array([0, 0, 0], dtype=np.uint8)
    upper_dark = np.array([15, 15, 15], dtype=np.uint8)

    # Define the slightly lighter color range (almost dark green)
    lower_light = np.array([0, 16, 0], dtype=np.uint8)
    upper_light = np.array([50, 120, 50], dtype=np.uint8)

    # Create masks for each color range
    dark_mask = cv2.inRange(image, lower_dark, upper_dark)
    light_mask = cv2.inRange(image, lower_light, upper_light)

    # Set pixels in the dark range to white
    image[dark_mask > 0] = [255, 255, 255]

    # Set pixels in the slightly lighter range to black
    image[light_mask > 0] = [0, 0, 0]

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the result
    cv2.imwrite(output_image_path, gray_img)

if __name__ == "__main__":
    input_path = "../dataset/floorplan_dataset/0.png"
    output_path = "./output_YAY/0.png"

    convert_colors(input_path, output_path)
