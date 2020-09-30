import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def diff_shape(test, true):
    test = test.copy()
    test[test!=0]=255
    diff = test - true
    dist = np.linalg.norm(diff)
    return dist

def match_rotation(test, true):
    true = true.copy()
    true[true!=0]=255
    min_diff = diff_shape(test, true)
    min_deg = None
    for deg in np.arange(1, 91, 0.5):
        r_test = rotate_image(test, deg)
        dist = diff_shape(r_test, true)
        print(dist)
        if (dist<min_diff):
            min_diff = dist
            min_deg = deg
        else: break

    if (min_deg is None):
        for deg in np.arange(1, 91, 0.5):
            r_test = rotate_image(test, -deg)
            dist = diff_shape(r_test, true)
            print(dist)
            if (dist<min_diff):
                min_diff = dist
                min_deg = -deg
            else: break
    
    return min_deg, min_diff
