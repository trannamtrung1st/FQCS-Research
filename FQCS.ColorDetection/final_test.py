from final_solution import color_detection as cd
import numpy as np 
import cv2 
import matplotlib.pyplot as plt

# -------------- PARAM ---------------------
# increase mean decrease sensitive ... (manual test)
# C1 = 6.5025
# C2 = 58.5225
C1 = 6.5025/3
C2 = 58.5225/3

psnrTriggerValue = 40
img_size = (32, 64)
blur_val = 0.05
alpha_r, alpha_l = 1, 1
beta_r, beta_l = -150, -150

# SEGMENT MATRIX
# matrix = (2, 2)
matrix = (4, 4)
# matrix = (8, 8)

# BIASES MATRIX
biases = np.array([
    [0.9, 1.1, 1, 0.8],
    [1, 1, 1, 0.8],
    [0.65, 1, 1, 0.65],
    [0.1, 1.1, 1.1, 0.1],
])

sat_adj = 2

min_similarity = 0.9
# -------------------------------------------
left = cv2.imread("d_left_3.jpg")
right = cv2.imread("d_right_3.jpg")
true_left = cv2.imread("d_left_3_true.jpg")
true_right = cv2.imread("d_right_3_true.jpg")

true_left = cd.preprocess(true_left, img_size, blur_val, alpha_l, beta_l, sat_adj)
true_right = cd.preprocess(true_right, img_size, blur_val, alpha_r, beta_r, sat_adj)

left_results, left_has_diff, right_results, right_has_diff = cd.detect_color_difference(left, right, true_left, true_right,
    biases, C1, C2, psnrTriggerValue, img_size, blur_val, alpha_r,beta_r,alpha_l,beta_l, sat_adj, matrix, min_similarity)

fig,axs = plt.subplots(1, 2)
left_res = "Left " if left_has_diff else ""
right_res = "Right " if right_has_diff else ""
if (left_has_diff or right_has_diff):
    plt.title("Different " + left_res + right_res)
axs[0].imshow(left)
axs[1].imshow(right)
plt.show()
