import numpy as np
import cv2
import matplotlib.pyplot as plt
import helper
import os
import detector


def main():
    os.chdir("FQCS")

    raw_cfg = detector.default_detector_config()
    raw_cfg["detect_method"] = "range"
    raw_cfg["d_cfg"] = detector.default_range_config()
    process_cfg = detector.preprocess_config(raw_cfg)

    uri = "test.mp4"
    cap = cv2.VideoCapture(uri)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

    found = False
    while not found:
        _, image = cap.read()
        image = cv2.resize(image, (640, 480))
        cv2.imshow("Original", image)
        cv2.waitKey(1)
        
        find_contours_func = detector.get_find_contours_func_by_method(process_cfg["detect_method"])
        pair = detector.detect_pair_and_size(
            image,
            find_contours_func,
            process_cfg['d_cfg'],
            min_area=process_cfg['min_area'],
            sample_area=process_cfg['sample_area'],
            stop_condition=process_cfg['stop_condition'],
            detect_range=process_cfg['detect_range'])

        if (pair is not None):
            found = True


if __name__ == "__main__":
    main()