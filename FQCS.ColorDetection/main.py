import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS_lib.FQCS import helper
from FQCS_lib.FQCS.manager import FQCSManager
from FQCS_lib.FQCS import detector
import os
import asyncio
import os


async def main():
    config_folder = "./"
    manager = FQCSManager(config_folder=config_folder)
    configs = manager.get_configs()
    main_cfg = None
    for cfg in configs:
        if cfg["is_main"] == True:
            main_cfg = cfg
            break
    if main_cfg is None: raise Exception("Invalid configuration")
    # main_cfg["detect_method"] = "edge"
    manager.load_sample_images()
    await manager.load_model(main_cfg)

    # uri = "test2.mp4"
    uri = main_cfg["camera_uri"]
    cap = cv2.VideoCapture(uri)

    while True:
        _, image = cap.read()
        resized_image, boxes, proc = manager.extract_boxes(main_cfg, image)

        # output
        cv2.imshow("Original", resized_image)

        final_grouped, sizes, check_group_idx, pair, split_left, split_right, image_detect = manager.detect_groups_and_checked_pair(
            main_cfg, boxes, resized_image)

        # output
        if image_detect is not None:
            cv2.imshow("Current detected", image_detect)

        # output
        unit = main_cfg["length_unit"]
        for idx, group in enumerate(final_grouped):
            for b_idx, b in enumerate(group):
                c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
                cur_size = sizes[idx][b_idx]
                lH, lW = cur_size
                helper.draw_boxes_and_sizes(resized_image, idx, box, lH, lW,
                                            unit, tl, br)
        cv2.imshow("Processed", resized_image)
        cv2.imshow("Contours processed", proc)
        cv2.waitKey(1)
        # cv2.waitKey(0)

        if (pair is not None):
            manager.check_group(check_group_idx, final_grouped)
            check_size = sizes[check_group_idx]
            h_diff, w_diff = manager.compare_size(main_cfg, check_size)

            if split_left is not None:
                # output
                plt.imshow(split_left)
                plt.show()
                plt.imshow(split_right)
                plt.show()

            # output
            left, right = pair
            left, right = left[0], right[0]
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(left)
            axs[0].set_title("Left detect")
            axs[1].imshow(right)
            axs[1].set_title("Right detect")
            if h_diff or w_diff:
                plt.title("Right detect: Different size")
            plt.show()

            left = cv2.flip(left, 1)
            sample_left_path, sample_right_path = manager.get_sample_path(
                True), manager.get_sample_path(False)
            if not os.path.exists(sample_left_path):
                cv2.imwrite(sample_left_path, left)
                cv2.imwrite(sample_right_path, right)
            else:
                pre_left, pre_right, pre_sample_left, pre_sample_right = manager.preprocess_images(
                    main_cfg, left, right)

                # Similarity compare
                sim_cfg = main_cfg["sim_cfg"]
                left_asym_task, right_asym_task = manager.detect_asym(
                    main_cfg, pre_left, pre_right, pre_sample_left,
                    pre_sample_right)

                images = [left, right]
                err_task = manager.detect_errors(main_cfg, images)

                left_color_task, right_color_task = manager.compare_colors(
                    main_cfg, pre_left, pre_right, pre_sample_left,
                    pre_sample_right)

                is_asym_diff_left, avg_asym_left, avg_amp_left, recalc_left, res_list_l, amp_res_list_l = await left_asym_task
                is_asym_diff_right, avg_asym_right, avg_amp_right, recalc_right, res_list_r, amp_res_list_r = await right_asym_task
                left_color_results = await left_color_task
                right_color_results = await right_color_task
                boxes, scores, classes, valid_detections = await err_task

                # output
                print("Min similarity: ", sim_cfg['min_similarity'])
                print("Left asymc: ", is_asym_diff_left, avg_asym_left,
                      avg_amp_left, recalc_left, res_list_l, amp_res_list_l)
                print("Right asymc: ", is_asym_diff_right, avg_asym_right,
                      avg_amp_right, recalc_right, res_list_r, amp_res_list_r)

                sample_left, sample_right = manager.get_sample_left(
                ), manager.get_sample_right()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(left)
                axs[0].set_title("Left detect")
                axs[1].imshow(sample_left)
                axs[1].set_title("Left sample")
                plt.show()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(right)
                axs[0].set_title("Right detect")
                axs[1].imshow(sample_right)
                axs[1].set_title("Right sample")
                plt.show()

                # output
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(pre_left)
                axs[1].imshow(pre_sample_left)
                plt.show()
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(pre_right)
                axs[1].imshow(pre_sample_right)
                plt.show()

                # output
                err_cfg = main_cfg["err_cfg"]
                helper.draw_yolo_results(
                    images,
                    boxes,
                    scores,
                    classes,
                    err_cfg["classes"],
                    err_cfg["img_size"],
                    min_score=err_cfg["yolo_score_threshold"])

                print("Left", left_color_results[1], left_color_results[2])
                print("Right", right_color_results[1], right_color_results[2])
                fig, axs = plt.subplots(1, 3)
                if (left_color_results[3]):
                    plt.title("Different left")
                axs[0].imshow(left)
                axs[1].imshow(sample_left)
                axs[2].imshow(images[0])
                plt.show()
                fig, axs = plt.subplots(1, 3)
                if (right_color_results[3]):
                    plt.title("Different right")
                axs[0].imshow(right)
                axs[1].imshow(sample_right)
                axs[2].imshow(images[1])
                plt.show()


def save_cfg():
    main_cfg = detector.default_detector_config()
    main_cfg["length_per_10px"] = 0.65
    main_cfg["color_cfg"]["amplify_thresh"] = (1000, 1000, 1000)
    side_cfg = detector.default_detector_config()
    side_cfg["is_main"] = False
    cfgs = [main_cfg, side_cfg]
    detector.save_json_cfg(cfgs, "./")


if __name__ == "__main__":
    asyncio.run(main())
    # save_cfg()