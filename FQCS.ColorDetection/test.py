import numpy as np
import cv2
import matplotlib.pyplot as plt
from FQCS_lib.FQCS import helper
from FQCS_lib.FQCS.manager import FQCSManager
from FQCS_lib.FQCS import detector
import os
import asyncio
import os

SAMPLE_AREA = 10000


async def main():
    config_folder = "./"
    detector_cfg = detector.load_json_cfg(config_folder)
    # detector_cfg["detect_method"] = "edge"
    manager = FQCSManager()

    uri = 0
    cap = cv2.VideoCapture(uri)
    frame_width, frame_height = detector_cfg["frame_width"], detector_cfg[
        "frame_height"]
    min_width, min_height = detector_cfg["min_width_per"], detector_cfg[
        "min_height_per"]
    min_width, min_height = frame_width * min_width, frame_height * min_height

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
            adj_bg_thresh = helper.adjust_thresh_by_brightness(
                image, d_cfg["light_adj_thresh"], d_cfg["bg_thresh"])
            d_cfg["adj_bg_thresh"] = adj_bg_thresh
        elif (detector_cfg["detect_method"] == "range"):
            adj_cr_to = helper.adjust_crange_by_brightness(
                image, d_cfg["light_adj_thresh"], d_cfg["cr_to"])
            d_cfg["adj_cr_to"] = adj_cr_to

        boxes, proc = detector.find_contours_and_box(
            image,
            find_contours_func,
            d_cfg,
            min_width=min_width,
            min_height=min_height,
            detect_range=detector_cfg['detect_range'])

        manager.group_pairs(boxes, SAMPLE_AREA)
        print("Checked:", manager.get_check_group())
        # pair, split_left, split_right = None, None, None
        # if manager.check_group < group_count:
        #     current_pair_boxes = grouped_pairs[manager.check_group]
        #     if current_pair_boxes is not None:
        #         image_detect = image.copy()
        #         pair, image_detect, split_left, split_right, current_pair_boxes = detector.detect_pair_and_size(
        #             image_detect,
        #             find_contours_func,
        #             d_cfg,
        #             current_pair_boxes,
        #             stop_condition=detector_cfg['stop_condition'])
        #         cv2.imshow("Current detected", image_detect)
        #         grouped_pairs[manager.check_group] = current_pair_boxes
        #     else:
        #         manager.checked_group(manager.check_group)

        # # output
        # unit = detector_cfg["length_unit"]
        # per_10px = detector_cfg["length_per_10px"]
        # sizes = []
        # for p in grouped_pairs:
        #     # if p is None or p[0] < manager.check_group: continue
        #     if p is None: continue
        #     for b in p[1:]:
        #         if b is None: continue
        #         c, rect, dimA, dimB, box, tl, tr, br, bl, minx, maxx, cenx = b
        #         _, _, min_x_group, max_x_group = manager.pos_tracks[p[0]]
        #         min_line = helper.extend_line((min_x_group, 0),
        #                                       (min_x_group, 1), 1000)
        #         max_line = helper.extend_line((max_x_group, 0),
        #                                       (max_x_group, 1), 1000)
        #         lH, lW = helper.calculate_length(
        #             dimA, per_10px), helper.calculate_length(dimB, per_10px)
        #         sizes.append((lH, lW))
        #         cv2.line(image, tuple(min_line[0]), tuple(min_line[1]),
        #                  (255, 0, 0))
        #         cv2.line(image, tuple(max_line[0]), tuple(max_line[1]),
        #                  (255, 0, 0))
        #         cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0),
        #                          2)
        #         cv2.putText(image, f"{p[0]}/ {lW:.1f} {unit}", (tl[0], tl[1]),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        #         cv2.putText(image, f"{lH:.1f} {unit}", (br[0], br[1]),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Processed", image)
        cv2.imshow("Contours processed", proc)
        if cv2.waitKey(0) == ord('c'):
            manager.check_group()


def save_cfg():
    detector_cfg = detector.default_detector_config()
    detector_cfg["length_per_10px"] = 0.65
    detector_cfg["color_cfg"]["amplify_thresh"] = (1000, 1000, 1000)
    detector.save_json_cfg(detector_cfg, "./")


if __name__ == "__main__":
    asyncio.run(main())
    # save_cfg()