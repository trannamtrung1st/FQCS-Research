import numpy as np
import cv2
import matplotlib.pyplot as plt
from detector import FQCSDetector
import helper
import os

# video and size capturing parameters
alpha = 1.0  # contrast control
beta = 0    # brightness control
threshold1 = 40  # canny control
threshold2 = 100  # canny control
kernel = (5, 5)  # init

true_left_path = "true_left.jpg"
true_right_path = "true_right.jpg"
os.chdir("FQCS_detector")
uri = "1.mp4"
cap = cv2.VideoCapture(uri)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)

# start
detector = FQCSDetector()

found = False
while not found:
    _,image = cap.read()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image[:,:,0] = 100
    # image[:,:,1] = 100
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imshow("Original", image)

    pair = detector.detect_pair_and_size(image=image,
        alpha=alpha,beta=beta,canny_threshold1=threshold1,canny_threshold2=threshold2,
        kernel=kernel,sigma_x=0,color_threshold=0, color_max_val=255,min_area=400,
        stop_condition=-50)
    
    if (pair is not None):
        found = True
        left,right = pair
        left,right=left[0],right[0]
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        # output
        fig,axs = plt.subplots(1, 2)
        axs[0].imshow(left)
        axs[1].imshow(right)
        plt.show()

        left = cv2.flip(left, 1)
        if not os.path.exists(true_left_path):
            cv2.imwrite(true_left_path, left)
            cv2.imwrite(true_right_path, right)
        else:
            true_left = cv2.imread(true_left_path)
            true_right = cv2.imread(true_right_path)
            
            # color detection parameters
            # increase mean decrease sensitive ... (manual test)
            # C1 = 6.5025
            # C2 = 58.5225
            C1 = 6.5025/3
            C2 = 58.5225/3
            psnrTriggerValue = 40
            img_size = (32, 64)
            blur_val = 0.03
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
            min_similarity = 0.8

            # output
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(left)
            axs[1].imshow(true_left)
            plt.show()
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(right)
            axs[1].imshow(true_right)
            plt.show()

            # start
            left = helper.match_rotation(left, true_left)
            right = helper.match_rotation(right, true_right)

            # output
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(left)
            axs[1].imshow(true_left)
            plt.show()
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(right)
            axs[1].imshow(true_right)
            plt.show()

            pre_true_left = detector.preprocess_for_cd(true_left, img_size, blur_val, alpha_l, beta_l, sat_adj)
            pre_true_right = detector.preprocess_for_cd(true_right, img_size, blur_val, alpha_r, beta_r, sat_adj)
            pre_left = detector.preprocess_for_cd(left, img_size, blur_val, alpha_l, beta_l, sat_adj)
            pre_right = detector.preprocess_for_cd(right, img_size, blur_val, alpha_r, beta_r, sat_adj)

            # output
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(pre_left)
            axs[1].imshow(pre_true_left)
            plt.show()
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(pre_right)
            axs[1].imshow(pre_true_right)
            plt.show()

            left_results, left_has_diff, right_results, right_has_diff = detector.detect_color_difference(pre_left, pre_right, pre_true_left, pre_true_right,
                biases, C1, C2, psnrTriggerValue, matrix, min_similarity)

            # output
            fig,axs = plt.subplots(1, 2)
            if (left_has_diff):
                plt.title("Different left")
            axs[0].imshow(left)
            axs[1].imshow(true_left)
            plt.show()
            fig,axs = plt.subplots(1, 2)
            if (right_has_diff):
                plt.title("Different right")
            axs[0].imshow(right)
            axs[1].imshow(true_right)
            plt.show()