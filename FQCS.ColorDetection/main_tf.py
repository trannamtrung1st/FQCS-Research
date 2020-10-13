import tensorflow as tf

from FQCS.tf2_yolov4.anchors import YOLOV4_ANCHORS
from FQCS.tf2_yolov4.model import YOLOv4
from FQCS.tf2_yolov4.convert_darknet_weights import convert_darknet_weights
import cv2
import numpy as np
import os

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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741]]

import matplotlib.pyplot as plt


def plot_results(pil_img, boxes, scores, classes):
    plt.imshow(pil_img)
    ax = plt.gca()

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(),
                                                   scores.tolist(),
                                                   classes.tolist()):
        if score > 0.5:
            ax.add_patch(
                plt.Rectangle((xmin, ymin),
                              xmax - xmin,
                              ymax - ymin,
                              fill=False,
                              color=COLORS[cl % 6],
                              linewidth=2))
            text = f'{CLASSES[cl]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=10)
    plt.axis('off')
    plt.show()


while True:
    img = cv2.imread("dirty_sorted/" + str(np.random.randint(151, 324)) +
                     ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    images = np.array([img]) / 255.
    boxes, scores, classes, valid_detections = model.predict(images)

    plot_results(
        images[0],
        boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT],
        scores[0],
        classes[0].astype(int),
    )
    if (cv2.waitKey() == ord('e')):
        break