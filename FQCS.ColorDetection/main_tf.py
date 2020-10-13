import tensorflow as tf

from FQCS.tf2_yolov4.anchors import YOLOV4_ANCHORS
from FQCS.tf2_yolov4.model import YOLOv4
from FQCS.tf2_yolov4.convert_darknet_weights import convert_darknet_weights
import cv2
import numpy as np
import os
from FQCS import detector
import asyncio


async def main():
    if not os.path.exists("./yolov4.h5"):
        convert_darknet_weights("./yolov4-custom_best.weights",
                                "./yolov4.h5", (416, 416, 3),
                                1,
                                weights=None)
        return

    yolo_cfg = detector.default_err_config()
    print("Test before")
    model = asyncio.create_task(
        detector.get_yolov4_model(
            inp_shape=yolo_cfg["inp_shape"],
            num_classes=yolo_cfg["num_classes"],
            training=False,
            yolo_max_boxes=yolo_cfg["yolo_max_boxes"],
            yolo_iou_threshold=yolo_cfg["yolo_iou_threshold"],
            weights=yolo_cfg["weights"],
            yolo_score_threshold=yolo_cfg["yolo_score_threshold"]))
    print("Test after")

    # COCO classes
    CLASSES = yolo_cfg["classes"]

    import matplotlib.pyplot as plt
    from FQCS.tf2_yolov4 import helper

    model = await model
    while True:
        img1 = cv2.imread("FQCS_detector/data/1/dirty_sorted/" +
                          str(np.random.randint(151, 324)) + ".jpg")
        img2 = cv2.imread("FQCS_detector/data/1/dirty_sorted/" +
                          str(np.random.randint(151, 324)) + ".jpg")
        images = [img1, img2]
        boxes, scores, classes, valid_detections = await asyncio.create_task(
            detector.detect_errors(model, images, yolo_cfg["img_size"]))
        helper.draw_results(images,
                            boxes,
                            scores,
                            classes,
                            CLASSES,
                            yolo_cfg["img_size"],
                            min_score=yolo_cfg["yolo_score_threshold"])
        cv2.imshow("Prediction", images[0])
        cv2.waitKey()
        cv2.imshow("Prediction", images[1])
        if (cv2.waitKey() == ord('e')):
            break


if __name__ == "__main__":
    asyncio.run(main())