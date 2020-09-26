import numpy as np 
import cv2 

img = cv2.imread('edge.jpg')
edges = cv2.Canny(img,50,50)
print(edges)

edges = cv2.resize(edges,(200, 200))
cv2.imshow("Test", edges)
cv2.waitKey(100000)