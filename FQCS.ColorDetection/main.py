import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS_lib.FQCS import helper
from FQCS_lib.FQCS.tf2_yolov4 import helper as y_helper
from FQCS_lib.FQCS.manager import FQCSManager
from FQCS_lib.FQCS import detector
import os
import asyncio
import os


async def main():
    config_folder = "./"
    sample_left_path = os.path.join(config_folder, detector.SAMPLE_LEFT_FILE)
    sample_right_path = os.path.join(config_folder, detector.SAMPLE_RIGHT_FILE)

    detector_cfg = detector.load_json_cfg(config_folder)
    # detector_cfg["detect_method"] = "edge"
    manager = FQCSManager()

    err_cfg = detector_cfg["err_cfg"]
    model = asyncio.create_task(
        detector.get_yolov4_model(
            inp_shape=err_cfg["inp_shape"],
            num_classes=err_cfg["num_classes"],
            training=False,
            yolo_max_boxes=err_cfg["yolo_max_boxes"],
            yolo_iou_threshold=err_cfg["yolo_iou_threshold"],
            weights=err_cfg["weights"],
            yolo_score_threshold=err_cfg["yolo_score_threshold"]))

    # uri = "test2.mp4"
    uri = 0
    cap = cv2.VideoCapture(uri)
    frame_width, frame_height = detector_cfg["frame_width"], detector_cfg[
        "frame_height"]
    min_width, min_height = detector_cfg["min_width_per"], detector_cfg[
        "min_height_per"]
    min_width, min_height = frame_width * min_width, frame_height * min_height
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

    sample_area = None
    sample_left, sample_right = None, None
    if os.path.exists(sample_left_path):
        sample_left = cv2.imread(sample_left_path)
        sample_right = cv2.imread(sample_right_path)
        sample_area = sample_left.shape[0] * sample_left.shape[1]

    model = await model
    while True:
        _, image = cap.read()
        image = cv2.resize(image, (frame_width, frame_height))

        # output
        cv2.imshow("Original", image)

        find_contours_func = detector.get_find_contours_func_by_method(
            detector_cfg["detect_method"])
        d_cfg = detector_cfg['d_cfg']

        # adjust thresh
        if (detector_cfg["detect_method"] == "thresh"):
            adj_thresh = d_cfg["light_adj_thresh"]
            if adj_thresh is not None and adj_thresh > 0:
                adj_bg_thresh = helper.adjust_thresh_by_brightness(
                    image, d_cfg["light_adj_thresh"], d_cfg["bg_thresh"])
            else:
                adj_bg_thresh = d_cfg["bg_thresh"]
            d_cfg["adj_bg_thresh"] = adj_bg_thresh
        elif (detector_cfg["detect_method"] == "range"):
            adj_thresh = d_cfg["light_adj_thresh"]
            if adj_thresh is not None and adj_thresh > 0:
                adj_cr_to = helper.adjust_crange_by_brightness(
                    image, d_cfg["light_adj_thresh"], d_cfg["cr_to"])
                d_cfg["adj_cr_to"] = adj_cr_to
            else:
                d_cfg["adj_cr_to"] = d_cfg["cr_to"]

        boxes, proc = detector.find_contours_and_box(
            image,
            find_contours_func,
            d_cfg,
            min_width=min_width,
            min_height=min_height,
            detect_range=detector_cfg['detect_range'])

        final_grouped, _, _, check_group_idx = manager.group_pairs(
            boxes, sample_area)
        group_count = manager.get_last_group_count()
        # print("Last min x:", manager.get_last_check_min_x())
        # print("Count:", group_count, "Check:", check_group_idx)

        pair, split_left, split_right = None, None, None
        check_group = None
        if check_group_idx is not None:
            check_group = final_grouped[check_group_idx]
            image_detect = image.copy()
            pair, image_detect, split_left, split_right, check_group = detector.detect_pair_and_size(
                image_detect,
                find_contours_func,
                d_cfg,
                check_group,
                stop_condition=detector_cfg['stop_condition'])
            cv2.imshow("Current detected", image_detect)
            final_grouped[check_group_idx] = check_group

        # output
        unit = detector_cfg["length_unit"]
        per_10px = detector_cfg["length_per_10px"]
        sizes = []
        for idx, group in enumerate(final_grouped):
            for b in group:
                c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
                lH, lW = helper.calculate_length(
                    dimA, per_10px), helper.calculate_length(dimB, per_10px)
                sizes.append((lH, lW))
                cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0),
                                 2)
                cv2.putText(image, f"{idx}/ {lW:.1f} {unit}", (tl[0], tl[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(image, f"{lH:.1f} {unit}", (br[0], br[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Processed", image)
        cv2.imshow("Contours processed", proc)
        cv2.waitKey(1)
        # cv2.waitKey(0)

        if (pair is not None):
            check_group_min_x = manager.get_min_x(check_group)
            manager.check_group(check_group_min_x)
            left, right = pair
            left, right = left[0], right[0]
            h_diff, w_diff = detector.compare_size(sizes[0], sizes[1],
                                                   detector_cfg)

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
            if h_diff or w_diff:
                plt.title("Right detect: Different size")
            plt.show()

            left = cv2.flip(left, 1)
            if not os.path.exists(sample_left_path):
                cv2.imwrite(sample_left_path, left)
                cv2.imwrite(sample_right_path, right)
            else:
                images = [left, right]
                err_task = asyncio.create_task(
                    detector.detect_errors(model, images, err_cfg["img_size"]))
                await asyncio.sleep(0)  # hacky way to trigger task

                # start
                c_cfg = detector_cfg['color_cfg']
                pre_sample_left = detector.preprocess_for_color_diff(
                    sample_left, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_l'], c_cfg['beta_l'], c_cfg['sat_adj'])
                pre_sample_right = detector.preprocess_for_color_diff(
                    sample_right, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_r'], c_cfg['beta_r'], c_cfg['sat_adj'])
                pre_left = detector.preprocess_for_color_diff(
                    left, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_l'], c_cfg['beta_l'], c_cfg['sat_adj'])
                pre_right = detector.preprocess_for_color_diff(
                    right, c_cfg['img_size'], c_cfg['blur_val'],
                    c_cfg['alpha_r'], c_cfg['beta_r'], c_cfg['sat_adj'])

                left_task, right_task = detector.detect_color_difference(
                    pre_left, pre_right, pre_sample_left, pre_sample_right,
                    c_cfg['amplify_thresh'], c_cfg['supp_thresh'],
                    c_cfg['amplify_rate'], c_cfg['max_diff'])

                # Similarity compare
                sim_cfg = detector_cfg["sim_cfg"]
                is_asym_diff_left, avg_asym_left, avg_amp_left, recalc_left, res_list_l, amp_res_list_l = (
                    await detector.detect_asym_diff(
                        pre_left, pre_sample_left, sim_cfg['segments_list'],
                        sim_cfg['C1'], sim_cfg['C2'], sim_cfg['psnr_trigger'],
                        sim_cfg['asym_amp_thresh'], sim_cfg['asym_amp_rate'],
                        sim_cfg['re_calc_factor_left'],
                        sim_cfg['min_similarity']))
                is_asym_diff_right, avg_asym_right, avg_amp_right, recalc_right, res_list_r, amp_res_list_r = (
                    await detector.detect_asym_diff(
                        pre_right, pre_sample_right, sim_cfg['segments_list'],
                        sim_cfg['C1'], sim_cfg['C2'], sim_cfg['psnr_trigger'],
                        sim_cfg['asym_amp_thresh'], sim_cfg['asym_amp_rate'],
                        sim_cfg['re_calc_factor_right'],
                        sim_cfg['min_similarity']))
                print("Min similarity: ", sim_cfg['min_similarity'])
                print("Left asymc: ", is_asym_diff_left, avg_asym_left,
                      avg_amp_left, recalc_left, res_list_l, amp_res_list_l)
                print("Right asymc: ", is_asym_diff_right, avg_asym_right,
                      avg_amp_right, recalc_right, res_list_r, amp_res_list_r)

                left_results = await left_task
                right_results = await right_task
                boxes, scores, classes, valid_detections = await err_task

                # output
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
                y_helper.draw_results(
                    images,
                    boxes,
                    scores,
                    classes,
                    err_cfg["classes"],
                    err_cfg["img_size"],
                    min_score=err_cfg["yolo_score_threshold"])

                print("Left", left_results[1], left_results[2])
                print("Right", right_results[1], right_results[2])
                fig, axs = plt.subplots(1, 3)
                if (left_results[3]):
                    plt.title("Different left")
                axs[0].imshow(left)
                axs[1].imshow(sample_left)
                axs[2].imshow(images[0])
                plt.show()
                fig, axs = plt.subplots(1, 3)
                if (right_results[3]):
                    plt.title("Different right")
                axs[0].imshow(right)
                axs[1].imshow(sample_right)
                axs[2].imshow(images[1])
                plt.show()


def save_cfg():
    detector_cfg = detector.default_detector_config()
    detector_cfg["length_per_10px"] = 0.65
    detector_cfg["color_cfg"]["amplify_thresh"] = (1000, 1000, 1000)
    detector.save_json_cfg(detector_cfg, "./")


if __name__ == "__main__":
    asyncio.run(main())
    # save_cfg()