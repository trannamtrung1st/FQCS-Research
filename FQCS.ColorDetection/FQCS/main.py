import numpy as np
import cv2
import matplotlib.pyplot as plt
import helper
import os
import detector


def main():
    os.chdir("FQCS")

    raw_cfg = detector.default_detector_config()
    raw_cfg["detect_method"] = "thresh"
    raw_cfg["d_cfg"] = detector.default_thresh_config()
    process_cfg = detector.preprocess_config(raw_cfg)

    uri = "test2.mp4"
    cap = cv2.VideoCapture(uri)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1100)

    found = False
    while not found:
        cv2.waitKey(1)
        _, image = cap.read()
        image = cv2.resize(image, (640, 480))

        # output
        cv2.imshow("Original", image)

        find_contours_func = detector.get_find_contours_func_by_method(
            process_cfg["detect_method"])
        pair, image, proc, boxes, split_left, split_right = detector.detect_pair_and_size(
            image,
            find_contours_func,
            process_cfg['d_cfg'],
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


if __name__ == "__main__":
    main()