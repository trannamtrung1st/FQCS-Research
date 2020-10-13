"""
Script to convert yolov4.weights file from AlexeyAB/darknet to tensorflow weights.

Initial implementation comes from https://github.com/zzh8829/yolov3-tf2
"""

import numpy as np

from .anchors import YOLOV4_ANCHORS
from .model import YOLOv4
from .weights import load_darknet_weights_in_yolo


def convert_darknet_weights(darknet_weights_path,
                            output_weights_path,
                            input_shape,
                            num_classes,
                            weights="darknet"):
    """ Converts yolov4 darknet weights to tensorflow weights (.h5 file)

    Args:
        darknet_weights_path (str): Input darknet weights filepath (*.weights).
        output_weights_path (str): Output tensorflow weights filepath (*.h5).
        num_classes (int): Number of output classes
    """
    model = YOLOv4(input_shape=input_shape,
                   num_classes=num_classes,
                   anchors=YOLOV4_ANCHORS,
                   weights=weights)
    # pylint: disable=E1101
    model.predict(np.random.random((1, *input_shape)))

    model = load_darknet_weights_in_yolo(model, darknet_weights_path)

    model.save(output_weights_path)
