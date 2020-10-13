import tensorflow as tf

from FQCS.tf2_yolov4.anchors import YOLOV4_ANCHORS
from FQCS.tf2_yolov4.model import YOLOv4
from FQCS.tf2_yolov4.convert_darknet_weights import convert_darknet_weights
import cv2
import numpy as np
import os


def main():
    if not os.path.exists("./yolov4.h5"):
        convert_darknet_weights("./yolov4-custom_best.weights",
                                "./yolov4.h5", (416, 416, 3),
                                1,
                                weights=None)
        return

    HEIGHT, WIDTH = (640, 320)
    model = YOLOv4(input_shape=(HEIGHT, WIDTH, 3),
                   anchors=YOLOV4_ANCHORS,
                   num_classes=1,
                   training=False,
                   yolo_max_boxes=100,
                   yolo_iou_threshold=0.5,
                   yolo_score_threshold=0.5,
                   weights=None)

    model.load_weights("./yolov4.h5")
    model.summary()

    # COCO classes
    CLASSES = ['dirt']

    import matplotlib.pyplot as plt
    from FQCS.tf2_yolov4 import helper

    while True:
        img = cv2.imread("FQCS_detector/data/1/dirty_sorted/" +
                         str(np.random.randint(151, 324)) + ".jpg")
        img = cv2.resize(img, (WIDTH, HEIGHT))
        images = np.array([img]) / 255.
        boxes, scores, classes, valid_detections = model.predict(images)

        helper.draw_results(images[0],
                            boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
                            scores[0], classes[0].astype(int), CLASSES)

        cv2.imshow("Prediction", images[0])
        if (cv2.waitKey() == ord('e')):
            break


if __name__ == "__main__":
    main()