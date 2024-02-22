import numpy as np 
import cv2 
import os
  
img_path = '../shapes/raw_images/quarter_circle.png'
out_path = '../shapes/raw/'
font = cv2.FONT_HERSHEY_COMPLEX 
img2 = cv2.imread(img_path, cv2.IMREAD_COLOR) 
  
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
  
height, width = img.shape


_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY) 
  

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 


if not os.path.exists(out_path):
    os.makedirs(out_path)

pts = []


for i, cnt in enumerate(contours):
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)


    n = approx.ravel()  
    for j in n:
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]

            pts.append((x,y))

  
        i += 1



f = open(f'{out_path}/quarter_circle.txt', 'w')
for point in pts:
    # Normalize (x,y) to (0,1)
    x = (np.double(point[0]) / width) 
    y = (np.double(point[1]) / height) 
    f.write(f'{x} {y}\n')
f.close()
  

