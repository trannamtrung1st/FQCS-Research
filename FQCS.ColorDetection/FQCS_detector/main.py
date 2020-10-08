import numpy as np
import cv2
import matplotlib.pyplot as plt
# from detector_edge import FQCSDetector
from detector_thresh import FQCSDetector
# from detector_range import FQCSDetector
import helper
import os

# video and size capturing parameters
alpha = 1.0  # contrast control
beta = 0    # brightness control
threshold1 = 40  # canny control
threshold2 = 100  # canny control
kernel = (5, 5)  # init
d_kernel = np.ones((5,5))
e_kernel = None        
light_adj_thresh = 65
bg_thresh = 110
cr_from,cr_to = (0,0,0), (180, 255*0.5, 255*0.5)

# color detection
img_size = (32, 64)
blur_val = 0.05
alpha_r, alpha_l = 1, 1
beta_r, beta_l = -150, -150
sat_adj = 2
supp_thresh = 10
amplify_thresh = (76,31,85)
amplify_rate = 20
max_diff = 0.2

true_left_path = "true_left.jpg"
true_right_path = "true_right.jpg"
os.chdir("FQCS_detector")
uri = "test1.mp4"
cap = cv2.VideoCapture(uri)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

# start
detector = FQCSDetector()

true_left,true_right,sample_area = None,None,None
if os.path.exists(true_left_path):
    true_left = cv2.imread(true_left_path)
    true_right = cv2.imread(true_right_path)
    sample_area = true_left.shape[0]*true_left.shape[1]

found = False
while not found:
    _,image = cap.read()
    image = cv2.resize(image, (640,480))
    # image = helper.change_contrast_and_brightness(image, 0.5, 0)
    # image = helper.rotate_image(image, 5)
    cv2.imshow("Original", image)

    # using edge 
    # pair = detector.detect_pair_and_size(image=image,
    #     alpha=alpha,beta=beta,canny_threshold1=threshold1,canny_threshold2=threshold2,
    #     kernel=kernel,d_kernel=d_kernel, e_kernel=e_kernel, sample_area=sample_area,
    #     stop_condition=100,detect_range=(0.05,0.95))

    # using thresh
    light_adj_val = helper.brightness(image)/light_adj_thresh
    adj_bg_thresh = bg_thresh * light_adj_val
    pair = detector.detect_pair_and_size(image=image,
        bg_thresh=adj_bg_thresh,
        sample_area=sample_area,
        stop_condition=0,detect_range=(0.2,0.8))

    # using range
    # light_adj_val = helper.brightness(image)/light_adj_thresh
    # adj_cr_to = (cr_to[0],cr_to[1]*light_adj_val,cr_to[2]*light_adj_val)
    # pair = detector.detect_pair_and_size(image=image,cr_from=cr_from,
    #     cr_to=adj_cr_to,sample_area=sample_area,stop_condition=0,detect_range=(0.2,0.8))
    
    if (pair is not None):
        found = True
        left,right = pair
        left,right=left[0],right[0]
        
        # output
        fig,axs = plt.subplots(1, 2)
        axs[0].imshow(left)
        axs[0].set_title("Left detect")
        axs[1].imshow(right)
        axs[1].set_title("Right detect")
        plt.show()

        left = cv2.flip(left, 1)
        if not os.path.exists(true_left_path):
            cv2.imwrite(true_left_path, left)
            cv2.imwrite(true_right_path, right)
        else:
            # output
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(left)
            axs[0].set_title("Left detect")
            axs[1].imshow(true_left)
            axs[1].set_title("Left sample")
            plt.show()
            fig,axs = plt.subplots(1, 2)
            axs[0].imshow(right)
            axs[0].set_title("Right detect")
            axs[1].imshow(true_right)
            axs[1].set_title("Right sample")
            plt.show()

            # start
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

            left_results, left_has_diff, right_results, right_has_diff = detector.detect_color_difference(
                pre_left, pre_right, pre_true_left, pre_true_right, amplify_thresh,supp_thresh, amplify_rate, max_diff)

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