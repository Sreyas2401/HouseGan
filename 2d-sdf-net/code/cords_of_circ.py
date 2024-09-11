import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = '../shapes/raw_images/hand_drawn.png'

out_path = '../shapes/raw/'

image = cv2.imread(img_path)

width, height, channels = image.shape

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = np.zeros_like(image)

with open(f'{out_path}/circle.txt', 'w') as f:
    if len(contours) > 0:
        for contour in contours:
            for point in contour:
                x = point[0][0] / width
                y = point[0][1] / height
                f.write(f'{x} {y}\n')
    else:
        print("No contours detected.")

