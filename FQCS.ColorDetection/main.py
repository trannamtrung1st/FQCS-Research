import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS import helper
import os
from FQCS import detector


def main():
    raw_cfg = detector.default_detector_config()
    # raw_cfg["detect_method"] = "thresh"
    # raw_cfg["d_cfg"] = detector.default_thresh_config()
    raw_cfg["color_cfg"]["amplify_thresh"] = (1000, 1000, 1000)
    process_cfg = detector.preprocess_config(raw_cfg)

    true_left_path = "true_left.jpg"
    true_right_path = "true_right.jpg"
    uri = "test.mp4"
    cap = cv2.VideoCapture(uri)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

    true_left, true_right = None, None
    if os.path.exists(true_left_path):
        true_left = cv2.imread(true_left_path)
        true_right = cv2.imread(true_right_path)
        raw_cfg['min_area'] = true_left.shape[0] * true_left.shape[1] * 0.25

    found = False
    while not found:
        _, image = cap.read()
        image = cv2.resize(image, (640, 480))

        # output
        cv2.imshow("Original", image)

        find_contours_func = detector.get_find_contours_func_by_method(
            process_cfg["detect_method"])
        d_cfg = process_cfg['d_cfg']

        # adjust thresh
        if (process_cfg["detect_method"] == "thresh"):
            adj_bg_thresh = helper.adjust_thresh_by_brightness(
                image, d_cfg["light_adj_thresh"], d_cfg["bg_thresh"])
            d_cfg["adj_bg_thresh"] = adj_bg_thresh
        elif (process_cfg["detect_method"] == "range"):
            adj_cr_to = helper.adjust_crange_by_brightness(
                image, d_cfg["light_adj_thresh"], d_cfg["cr_to"])
            d_cfg["adj_cr_to"] = adj_cr_to

        pair, image, proc, boxes, split_left, split_right = detector.detect_pair_and_size(
            image,
            find_contours_func,
            d_cfg,
            min_area=process_cfg['min_area'],
            stop_condition=process_cfg['stop_condition'],
            detect_range=process_cfg['detect_range'])

        # output
        for b in boxes:
            dimA, dimB, box, tl, tr, br, bl = b
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
            cv2.putText(image, "{:.1f}px".format(dimA), (tl[0], tl[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(image, "{:.1f}px".format(dimB), (br[0], br[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Processed", image)
        cv2.imshow("Contours processed", proc)
        cv2.waitKey(1)

        if (pair is not None):
            found = True
            left, right = pair
            left, right = left[0], right[0]

            if split_left is not None:
                # output
                plt.imshow(split_left)
                plt.show()
                plt.imshow(split_right)
                plt.show()

            # output
            fig, axs = plt.subplots(1, 2)
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
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(left)
                axs[0].set_title("Left detect")
                axs[1].imshow(true_left)
                axs[1].set_title("Left sample")
                plt.show()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(right)
                axs[0].set_title("Right detect")
                axs[1].imshow(true_right)
                axs[1].set_title("Right sample")
                plt.show()

                # start
                c_cfg = process_cfg['color_cfg']
                pre_true_left = detector.preprocess_for_color_diff(
                    true_left, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_l'], c_cfg['beta_l'], c_cfg['sat_adj'])
                pre_true_right = detector.preprocess_for_color_diff(
                    true_right, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_r'], c_cfg['beta_r'], c_cfg['sat_adj'])
                pre_left = detector.preprocess_for_color_diff(
                    left, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_l'], c_cfg['beta_l'], c_cfg['sat_adj'])
                pre_right = detector.preprocess_for_color_diff(
                    right, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_r'], c_cfg['beta_r'], c_cfg['sat_adj'])

                # output
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(pre_left)
                axs[1].imshow(pre_true_left)
                plt.show()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(pre_right)
                axs[1].imshow(pre_true_right)
                plt.show()

                left_results, right_results = detector.detect_color_difference(
                    pre_left, pre_right, pre_true_left, pre_true_right,
                    c_cfg['amplify_thresh'], c_cfg['supp_thresh'],
                    c_cfg['amplify_rate'], c_cfg['max_diff'])

                # output
                print("Left", left_results[1], left_results[2])
                print("Right", right_results[1], right_results[2])
                fig, axs = plt.subplots(1, 2)
                if (left_results[3]):
                    plt.title("Different left")
                axs[0].imshow(left)
                axs[1].imshow(true_left)
                plt.show()
                fig, axs = plt.subplots(1, 2)
                if (right_results[3]):
                    plt.title("Different right")
                axs[0].imshow(right)
                axs[1].imshow(true_right)
                plt.show()


if __name__ == "__main__":
    main()