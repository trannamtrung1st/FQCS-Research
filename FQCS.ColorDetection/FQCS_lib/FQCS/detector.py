import numpy as np
import cv2
from . import helper
import imutils
import asyncio
from .tf2_yolov4.anchors import YOLOV4_ANCHORS
from .tf2_yolov4.model import YOLOv4
import json
import os

CONFIG_FILE = "config.json"
SAMPLE_LEFT_FILE = "sample_left.jpg"
SAMPLE_RIGHT_FILE = "sample_right.jpg"


def compare_size(lsize, rsize, detector_cfg):
    max_size_diff = detector_cfg["max_size_diff"]
    return abs(lsize[0] -
               rsize[0]) >= max_size_diff, abs(lsize[1] -
                                               rsize[1]) >= max_size_diff


async def get_yolov4_model(inp_shape=(320, 160, 3),
                           num_classes=1,
                           training=False,
                           yolo_max_boxes=10,
                           yolo_iou_threshold=0.5,
                           yolo_score_threshold=0.5,
                           weights="yolov4.h5"):
    model = YOLOv4(input_shape=inp_shape,
                   anchors=YOLOV4_ANCHORS,
                   num_classes=num_classes,
                   training=training,
                   yolo_max_boxes=yolo_max_boxes,
                   yolo_iou_threshold=yolo_iou_threshold,
                   yolo_score_threshold=yolo_score_threshold,
                   weights=None)
    model.load_weights(weights)
    return model


async def detect_errors(model, images, img_size):
    for i in range(len(images)):
        img = cv2.resize(images[i], img_size)
        images[i] = img / 255.
    images = np.asarray(images)
    boxes, scores, classes, valid_detections = model.predict(images)
    return boxes, scores, classes, valid_detections


def get_find_contours_func_by_method(m_name):
    if m_name == "edge":
        return find_contours_using_edge
    if m_name == "thresh":
        return find_contours_using_thresh
    if m_name == "range":
        return find_contours_using_range


def default_err_config():
    return dict(inp_shape=(320, 160, 3),
                img_size=(160, 320),
                num_classes=1,
                yolo_max_boxes=10,
                yolo_iou_threshold=0.5,
                weights="yolov4.h5",
                yolo_score_threshold=0.3,
                classes=['dirt'])


def default_edge_config():
    return dict(alpha=1.0,
                beta=0,
                threshold1=40,
                threshold2=100,
                kernel=(5, 5),
                d_kernel=np.ones((5, 5)),
                e_kernel=None)


def default_thresh_config():
    return dict(bg_thresh=110, adj_bg_thresh=110, light_adj_thresh=65)


def default_range_config():
    cr_to = (180, 255 * 0.5, 255 * 0.5)
    return dict(cr_from=(0, 0, 0),
                cr_to=cr_to,
                adj_cr_to=cr_to,
                light_adj_thresh=65)


def default_d_config():
    cr_to = (180, 255 * 0.5, 255 * 0.5)
    return dict(bg_thresh=110,
                adj_bg_thresh=110,
                light_adj_thresh=65,
                alpha=1.0,
                beta=0,
                threshold1=40,
                threshold2=100,
                kernel=(5, 5),
                d_kernel=np.ones((5, 5)),
                e_kernel=None,
                cr_from=(0, 0, 0),
                cr_to=cr_to,
                adj_cr_to=cr_to)


def default_color_config():
    return dict(img_size=(32, 64),
                blur_val=0.05,
                alpha_r=1,
                alpha_l=1,
                beta_r=-150,
                beta_l=-150,
                sat_adj=2,
                supp_thresh=10,
                amplify_thresh=(None, None, None),
                amplify_rate=20,
                max_diff=0.2)


def default_detector_config():
    color_cfg = default_color_config()
    err_cfg = default_err_config()
    d_cfg = default_d_config()
    detector_config = dict(min_width_per=0.1,
                           min_height_per=0.7,
                           stop_condition=0,
                           detect_range=(0.2, 0.8),
                           length_per_10px=None,
                           length_unit="cm",
                           max_size_diff=0.3,
                           frame_width=640,
                           frame_height=480,
                           color_cfg=color_cfg,
                           detect_method="thresh",
                           err_cfg=err_cfg,
                           d_cfg=d_cfg)
    return detector_config


def save_json_cfg(cfg, folder_path):
    cfg_path = os.path.join(folder_path, CONFIG_FILE)
    if (cfg['detect_method'] == "edge"):
        d_kernel = cfg["d_cfg"]["d_kernel"]
        e_kernel = cfg["d_cfg"]["e_kernel"]
        if d_kernel is not None:
            cfg["d_cfg"]["d_kernel"] = d_kernel.shape
        if e_kernel is not None:
            cfg["d_cfg"]["e_kernel"] = e_kernel.shape
    with open(cfg_path, 'w') as fo:
        json.dump(cfg, fo, indent=2)


def load_json_cfg(folder_path):
    cfg_path = os.path.join(folder_path, CONFIG_FILE)
    with open(cfg_path) as fi:
        cfg = json.load(fi)
        if (cfg['detect_method'] == "edge"):
            kernel = cfg['d_cfg']['kernel']
            kernel = (kernel[0], kernel[1])
            cfg['d_cfg']['kernel'] = kernel
            d_kernel = cfg['d_cfg']['d_kernel']
            e_kernel = cfg['d_cfg']['e_kernel']
            if d_kernel is not None:
                d_kernel = np.ones((d_kernel[0], d_kernel[1]))
                cfg['d_cfg']['d_kernel'] = d_kernel
            if e_kernel is not None:
                e_kernel = np.ones((e_kernel[0], e_kernel[1]))
                cfg['d_cfg']['e_kernel'] = e_kernel
        elif (cfg['detect_method'] == "range"):
            cr_from = cfg['d_cfg']['cr_from']
            cr_to = cfg['d_cfg']['cr_to']
            cr_from = (cr_from[0], cr_from[1], cr_from[2])
            cr_to = (cr_to[0], cr_to[1], cr_to[2])
            cfg['d_cfg']['cr_from'] = cr_from
            cfg['d_cfg']['cr_to'] = cr_to
            cfg['d_cfg']['adj_cr_to'] = cr_to
        elif (cfg['detect_method'] == 'thresh'):
            bg_thresh = cfg['d_cfg']['bg_thresh']
            cfg['d_cfg']['adj_bg_thresh'] = bg_thresh

        detect_range = cfg['detect_range']
        detect_range = (detect_range[0], detect_range[1])
        img_size = cfg['color_cfg']['img_size']
        img_size = (img_size[0], img_size[1])
        amplify_thresh = cfg['color_cfg']['amplify_thresh']
        amplify_thresh = (amplify_thresh[0], amplify_thresh[1],
                          amplify_thresh[2])
        cfg['detect_range'] = detect_range
        cfg['color_cfg']['amplify_thresh'] = amplify_thresh
        cfg['color_cfg']['img_size'] = img_size

        err_cfg = cfg["err_cfg"]
        img_size = err_cfg["img_size"]
        inp_shape = err_cfg["inp_shape"]
        img_size = (img_size[0], img_size[1])
        inp_shape = (inp_shape[0], inp_shape[1], inp_shape[2])
        err_cfg['img_size'] = img_size
        err_cfg['inp_shape'] = inp_shape

    return cfg


def preprocess_for_color_diff(img,
                              img_size=(32, 64),
                              blur_val=0.05,
                              alpha: float=1,
                              beta=-150,
                              sat_adj=2):
    if (sat_adj != 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= sat_adj
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    img = helper.change_contrast_and_brightness(img, alpha, beta)
    ksize_w = round(img.shape[0] * blur_val)
    ksize_w = ksize_w if ksize_w % 2 == 1 else ksize_w + 1
    ksize_h = round(img.shape[1] * blur_val)
    ksize_h = ksize_h if ksize_h % 2 == 1 else ksize_h + 1
    
    if blur_val is not None:
        img = cv2.blur(
            img,
            (ksize_w,ksize_h))
    img = cv2.resize(img, img_size)
    return img


async def find_color_diff(test, true, amplify_thresh, supp_thresh,
                          amplify_rate, max_diff):
    test_hist = helper.get_hist_bgr(test)
    true_hist = helper.get_hist_bgr(true)
    list_dist = np.zeros((3, ))
    w, h, _ = true.shape
    max_dist = w * h
    for i in range(3):
        diff = np.abs(test_hist[i] - true_hist[i])
        diff[diff < supp_thresh] = 0
        dist = np.linalg.norm(diff)
        if (dist > amplify_thresh[i]):
            dist *= (dist / amplify_thresh[i])**amplify_rate
        list_dist[i] = dist
    sum_dist = np.sum(list_dist)
    avg = sum_dist / max_dist
    return sum_dist, avg, list_dist, avg >= max_diff


def detect_color_difference(left,
                            right,
                            true_left,
                            true_right,
                            amplify_thresh=None,
                            supp_thresh=None,
                            amplify_rate=None,
                            max_diff=None):
    # START
    left_task = asyncio.create_task(
        find_color_diff(left, true_left, amplify_thresh, supp_thresh,
                        amplify_rate, max_diff))
    right_task = asyncio.create_task(
        find_color_diff(right, true_right, amplify_thresh, supp_thresh,
                        amplify_rate, max_diff))
    return left_task, right_task


def find_contours_using_edge(image, d_cfg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, d_cfg['kernel'], 0)
    enhanced = cv2.convertScaleAbs(gray,
                                   alpha=d_cfg['alpha'],
                                   beta=d_cfg['beta'])
    edged = cv2.Canny(enhanced, d_cfg['threshold1'], d_cfg['threshold2'])
    edged = cv2.dilate(edged, d_cfg['d_kernel'], iterations=1)
    edged = cv2.erode(edged, d_cfg['e_kernel'], iterations=1)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts, edged


def find_contours_using_range(image, d_cfg):
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame, d_cfg['cr_from'], d_cfg['adj_cr_to'])
    h, w, _ = image.shape
    im_th = np.zeros((h, w), dtype="ubyte")
    im_th[mask < 127] = 255
    cnts = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts, im_th


def find_contours_using_thresh(image, d_cfg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, d_cfg['adj_bg_thresh'], 255,
                                cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts, thresh


def find_contours_and_box(image: np.ndarray, find_contours_func, d_cfg,
                          min_width, min_height):
    # start
    boxes = []
    cnts, proc = find_contours_func(image, d_cfg)
    helper.fill_contours(image, cnts)
    for c in cnts[:2]:
        rect, dimA, dimB, box, tl, tr, br, bl = helper.find_cnt_box(c, image)
        if (dimA >= min_height and dimB >= min_width):
            boxes.append((rect, dimA, dimB, box, tl, tr, br, bl))
    return boxes, cnts, proc


def detect_one_and_size(orig_img: np.ndarray, image: np.ndarray,
                        find_contours_func, d_cfg):
    # start
    h, w = image.shape[:2]
    cnts, proc = find_contours_func(image, d_cfg)
    helper.fill_contours(image, cnts)
    c = cnts[0]
    rect, dimA, dimB, box, tl, tr, br, bl = helper.find_cnt_box(c, image)
    warped = helper.get_warped_box(image, rect, box)
    return (warped, (rect, dimA, dimB, box, tl, tr, br, bl))


def detect_pair_and_size(image: np.ndarray,
                         find_contours_func,
                         d_cfg,
                         boxes,
                         cnts,
                         stop_condition=0,
                         detect_range=(0.2, 0.8)):
    # start
    pair = []
    h, w = image.shape[:2]
    min_x, max_x = w, 0
    from_x, to_x = w * detect_range[0], w * detect_range[1]
    for item in boxes:
        rect, dimA, dimB, box, tl, tr, br, bl = item
        cur_min_x = min(tl[0], tr[0], br[0], bl[0])
        cur_max_x = max(tl[0], tr[0], br[0], bl[0])
        min_x = min(cur_min_x, min_x)
        max_x = max(cur_max_x, max_x)
        if (min_x < from_x or max_x > to_x):
            break

        warped = helper.get_warped_box(image, rect, box)
        pair.append((warped, box, dimA, dimB))

    center_val = w - max_x - min_x
    is_center = True if (center_val <= stop_condition) else False
    if (not is_center):
        pair = []

    split_left, split_right = None, None
    if (len(pair) == 1):
        split_left, split_right = split_pair(image, cnts[0])
        if (split_left is not None):
            left = detect_one_and_size(image, split_left, find_contours_func,
                                       d_cfg)
            right = detect_one_and_size(image, split_right, find_contours_func,
                                        d_cfg)
            if (left is not None and right is not None):
                pair = [left, right]
                boxes = [left[1], right[1]]

    pair = sorted(pair, key=lambda x: x[1][0][0], reverse=True)
    return pair if len(
        pair) == 2 else None, image, split_left, split_right, boxes


def split_pair(img, cnt):
    h, w, _ = img.shape
    hull = cv2.convexHull(cnt, returnPoints=False)
    hull = sorted(hull, reverse=True)
    hull = np.array(hull)
    defects = cv2.convexityDefects(cnt, hull)
    if (defects is None or len(defects) < 2):
        return None, None
    defects = sorted(defects, key=lambda x: x[0][3], reverse=True)
    defects = np.array(defects)
    fars = []
    for i in range(2):
        s, e, f, d = defects[i][0]
        far = tuple(cnt[f][0])
        fars.append(far)
    if (fars[0] == fars[1]):
        fars = np.array(fars)
        fars[0][1] -= 1
    fars = sorted(fars, key=lambda x: x[1])

    p1, p2 = helper.extend_line(fars[0], fars[1], 1000)
    pts_r = np.array([0, 0, p1[0], p1[1], p2[0], p2[1], 0, h])
    pts_l = np.array([w, 0, w, h, p2[0], p2[1], p1[0], p1[1]])
    pts_r = pts_r.reshape((-1, 1, 2))
    pts_l = pts_l.reshape((-1, 1, 2))

    left = cv2.fillPoly(img.copy(), [pts_l], (0, 0, 0))
    right = cv2.fillPoly(img.copy(), [pts_r], (0, 0, 0))
    return left, right