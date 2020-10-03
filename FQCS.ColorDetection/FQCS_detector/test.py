import numpy as np 
import cv2 
import os
import matplotlib.pyplot as plt

os.chdir("FQCS_detector")

def get_hist(uri):
    img = cv2.imread(uri)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    return hist_h, hist_s, hist_v

left_hist = get_hist("true_right.jpg")
right_hist = get_hist("test_right.jpg")

# output
for i in range(3):
    fig,axs = plt.subplots(1, 2)
    axs[0].plot(left_hist[i])
    axs[1].plot(right_hist[i])
    plt.show()