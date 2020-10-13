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

    inp_shape = (640, 320, 3)
    img_size = (320, 640)
    model = detector.get_yolov4_model(inp_shape=inp_shape,
                                      yolo_score_threshold=0.3)
    model.summary()

    # COCO classes
    CLASSES = ['dirt']

    import matplotlib.pyplot as plt
    from FQCS.tf2_yolov4 import helper

    while True:
        img1 = cv2.imread("FQCS_detector/data/1/dirty_sorted/" +
                          str(np.random.randint(151, 324)) + ".jpg")
        img2 = cv2.imread("FQCS_detector/data/1/dirty_sorted/" +
                          str(np.random.randint(151, 324)) + ".jpg")
        images = [img1, img2]
        boxes, scores, classes, valid_detections = await asyncio.create_task(
            detector.detect_errors(model, images, img_size))
        helper.draw_results(images,
                            boxes,
                            scores,
                            classes,
                            CLASSES,
                            img_size,
                            min_score=0.3)
        cv2.imshow("Prediction", images[0])
        cv2.waitKey()
        cv2.imshow("Prediction", images[1])
        if (cv2.waitKey() == ord('e')):
            break


if __name__ == "__main__":
    asyncio.run(main())