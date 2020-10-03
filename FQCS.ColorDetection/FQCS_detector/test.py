import numpy as np 
import cv2 
import os
import matplotlib.pyplot as plt

os.chdir("FQCS_detector")

left1 = cv2.imread("true_right_1.jpg")
left2 = cv2.imread("true_right.jpg")
# left1 = cv2.imread("true_left_1.jpg")
# left2 = cv2.imread("true_left.jpg")
w,h,_ = left1.shape
left2 = cv2.resize(left2, (h,w))
left1 = cv2.blur(left1, (10, 10))
left2 = cv2.blur(left2, (10, 10))
cv2.imshow("test", left1)
cv2.waitKey(1000000)
cv2.imshow("test", left2)
cv2.waitKey(1000000)

diff = left1-left2
diff[diff<0]=0
cv2.imshow("test", diff)
cv2.waitKey(1000000)