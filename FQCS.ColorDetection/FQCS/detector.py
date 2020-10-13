from scipy.spatial import distance as dist
import numpy as np
import cv2
import helper
import imutils
from imutils import perspective
from easydict import EasyDict as edict


def preprocess_for_color_diff(img,
                              img_size=(32, 64),
                              blur_val=0.05,
                              alpha=1,
                              beta=-150,
                              sat_adj=2):
    if (sat_adj != 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= sat_adj
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = helper.change_contrast_and_brightness(img, alpha, beta)
    if blur_val is not None:
        img = cv2.blur(
            img,
            (round(img.shape[0] * blur_val), round(img.shape[1] * blur_val)))
    img = cv2.resize(img, img_size)
    return img


def find_color_diff(test, true, amp_thresh, supp_thresh, amplify_rate,
                    max_diff):
    test_hist = helper.get_hist_bgr(test)
    true_hist = helper.get_hist_bgr(true)
    list_dist = np.zeros((3, ))
    w, h, _ = true.shape
    max_dist = w * h
    for i in range(3):
        diff = np.abs(test_hist[i] - true_hist[i])
        diff[diff < supp_thresh] = 0
        dist = np.linalg.norm(diff)
        if (dist > amp_thresh[i]):
            dist *= (dist / amp_thresh[i])**amplify_rate
        list_dist[i] = dist
    sum_dist = np.sum(list_dist)
    avg = sum_dist / max_dist
    return sum_dist, avg >= max_diff


def find_contours_using_edge(image, d_cfg):
    cfg = edict(d_cfg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, cfg.kernel, 0)
    enhanced = cv2.convertScaleAbs(gray, alpha=cfg.alpha, beta=cfg.beta)
    edged = cv2.Canny(enhanced, cfg.canny_threshold1, cfg.canny_threshold2)
    edged = cv2.dilate(edged, cfg.d_kernel, iterations=1)
    edged = cv2.erode(edged, cfg.e_kernel, iterations=1)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts


def find_contours_using_range(self, image, d_cfg):
    cfg = edict(d_cfg)
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvFrame, cfg.cr_from, cfg.cr_to)
    h, w, _ = image.shape
    im_th = np.zeros((h, w), dtype="ubyte")
    im_th[mask < 127] = 255
    cnts = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts


def find_contours_using_thresh(self, image, d_cfg):
    cfg = edict(d_cfg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, cfg.bg_thresh, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    return cnts


def detect_one_and_size(orig_img: np.ndarray, image: np.ndarray,
                        find_contours_func, d_cfg):
    # start
    h, w = image.shape[:2]
    cnts = find_contours_func(image, d_cfg)
    helper.fill_contours(image, cnts)
    c = cnts[0]
    rect, dimA, dimB, box, tl, tr, br, bl = helper.find_cnt_box(c, image)
    warped = get_warped_cnt(image, rect, box)
    return (warped, box, dimA, dimB)


def detect_pair_and_size(image: np.ndarray,
                         find_contours_func,
                         d_cfg,
                         min_area=None,
                         sample_area=None,
                         stop_condition=0,
                         detect_range=(0.2, 0.8)):
    # start
    pair = []
    h, w = image.shape[:2]
    cnts = find_contours_func(image, d_cfg)
    helper.fill_contours(image, cnts)
    min_x, max_x = w, 0
    from_x, to_x = w * detect_range[0], w * detect_range[1]
    for c in cnts[:2]:
        if cv2.contourArea(c) < min_area:
            break
        rect, dimA, dimB, box, tl, tr, br, bl = find_cnt_box(c, orig)
        cur_min_x = min(tl[0], tr[0], br[0], bl[0])
        cur_max_x = max(tl[0], tr[0], br[0], bl[0])
        min_x = min(cur_min_x, min_x)
        max_x = max(cur_max_x, max_x)
        if (min_x < from_x or max_x > to_x):
            break

        center_val = w - max_x - min_x
        is_center = True if (center_val <= stop_condition) else False
        if (is_center):
            warped = helper.get_warped_box(image, rect, box)
            pair.append((warped, box, dimA, dimB))

    split_left, split_right = None, None
    if (sample_area is not None and len(pair) == 1):
        area = pair[0][2] * pair[0][3]
        if (area >= sample_area * 1.25):
            split_left, split_right = split_pair(image, cnts[0])
    if (split_left is not None):
        left = detect_one_and_size(image, split_left, find_contours_func,
                                   d_cfg)
        right = detect_one_and_size(image, split_right, find_contours_func,
                                    d_cfg)
        if (left is not None and right is not None):
            pair = [left, right]

    pair = sorted(pair, key=lambda x: x[1][0][0], reverse=True)
    return pair if len(pair) == 2 else None


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