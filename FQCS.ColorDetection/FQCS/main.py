import numpy as np
import cv2
import matplotlib.pyplot as plt
import helper
import os
from easydict import EasyDict as edict
import detector


def main():
    os.chdir("FQCS")
    edge_cfg = edict(alpha=1.0,
                     beta=0,
                     threshold1=40,
                     threshold2=100,
                     kernel=[5, 5],
                     d_kernel=[5, 5],
                     e_kernel=None)
    light_adj_thresh = 65
    thresh_cfg = edict(bg_thresh=110)
    range_cfg = edict(cr_from=(0, 0, 0), cr_to=(180, 255 * 0.5, 255 * 0.5))

    color_cfg = edict(img_size=(32, 64),
                      blur_val=0.05,
                      alpha_r=1,
                      alpha_l=1,
                      beta_r=-150,
                      beta_l=-150,
                      sat_adj=2,
                      supp_thresh=10,
                      amplify_thresh=(76, 31, 85),
                      amplify_rate=20,
                      max_diff=0.2)

    true_left_path = "true_left.jpg"
    true_right_path = "true_right.jpg"
    uri = "test.mp4"
    cap = cv2.VideoCapture(uri)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

    found = False
    while not found:
        _, image = cap.read()
        image = cv2.resize(image, (640, 480))
        cv2.imshow("Original", image)
        cv2.waitKey(10)

        if (pair == None):
            found = True


if __name__ == "__main__":
    main()