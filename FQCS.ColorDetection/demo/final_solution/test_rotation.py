import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shape_rotation as r

os.chdir("final_solution")
left = cv2.imread("left.jpg")
left_true = cv2.imread("left_true.jpg")
min_deg, min_diff = r.match_rotation(left, left_true)
print(min_diff, min_deg)
r_left = r.rotate_image(left, min_deg)
diff = r_left - left_true
cv2.imshow("test", diff)
cv2.waitKey(1000000)
