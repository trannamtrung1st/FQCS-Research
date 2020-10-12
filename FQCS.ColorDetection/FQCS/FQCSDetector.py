from scipy.spatial import distance as dist
import numpy as np
import cv2
import helper
import config
import imutils
from imutils import perspective

class FQCSDetector:

    def __init__(self):
        self.color_detection_cfg = config.ColorDetectionConfig()
        self.err_detection_cfg = config.ErrorDetectionConfig()
        self.true_left = None
        self.true_right = None
        self.min_area = None
        self.stop_condition = 0
        self.detect_range = (0.2,0.8)
    
    def preprocess_for_color_detection(self, img, alpha, beta):
        c = self.color_detection_cfg
        if (c.saturation_adj!=1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:,:,1]*=c.saturation_adj
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = helper.change_contrast_and_brightness(img, alpha, beta)
        if c.blur_val is not None:
            img = cv2.blur(img, (round(img.shape[0] * c.blur_val),round(img.shape[1] * c.blur_val)))
        img = cv2.resize(img, c.img_size)
        return img

    def _find_color_difference(self, test, true):
        c = self.color_detection_cfg
        test_hist = helper.get_hist_bgr(test)
        true_hist = helper.get_hist_bgr(true)
        list_dist = np.zeros((3,))
        w,h,_= true.shape
        max_dist = w*h
        for i in range(3):
            diff = np.abs(test_hist[i]-true_hist[i])
            diff[diff< c.suppress_thresh]=0
            dist = np.linalg.norm(diff)
            if (dist> c.amplify_thresh[i]):
                dist*=(dist/c.amplify_thresh[i])**c.amplify_rate
            list_dist[i] = dist
        sum_dist = np.sum(list_dist)
        avg = sum_dist/max_dist
        return sum_dist, avg>=max_diff

    def detect_color_difference(self, left, right):
        left_results, left_has_diff = self._find_color_difference(left, self.true_left)
        right_results, right_has_diff = self._find_color_difference(right, self.true_right)
        return left_results, left_has_diff, right_results, right_has_diff

    def _detect_one_and_size(self, orig_img: np.ndarray, image: np.ndarray):
        # start
        h, w = image.shape[:2]
        cnts = self.find_contours(image)
        helper.fill_countours(image, cnts)
        c = cnts[0]
        rect,dimA,dimB,box,tl,tr,br,bl = helper.find_contours_box(c, image)
        warped = helper.get_warped_box(image, rect, box)
        return (warped,box,dimA,dimB)

    def detect_pair_and_size(self, image: np.ndarray):
        # start
        pair = []
        h, w = image.shape[:2]
        cnts = self.find_contours(image)
        helper.fill_countours(image, cnts)
        min_x,max_x = w,0
        from_x,to_x = w*detect_range[0],w*detect_range[1]
        for c in cnts[:2]:
            if cv2.contourArea(c) < self.min_area:
                break
            rect,dimA,dimB,box,tl,tr,br,bl = helper.find_contours_box(c, image)
            cur_min_x = min(tl[0], tr[0], br[0], bl[0])
            cur_max_x = max(tl[0], tr[0], br[0], bl[0])
            min_x = min(cur_min_x, min_x)
            max_x = max(cur_max_x, max_x)
            if (min_x<from_x or max_x>to_x):
                break

            center_val = w-max_x-min_x
            is_center = True if (center_val<=stop_condition) else False
            if (is_center):
                warped = self.get_warped_cnt(image, rect, box)
                pair.append((warped,box,dimA,dimB))
        
        split_left, split_right = None, None
        if (self.min_area is not None and len(pair)==1):
            area = pair[0][2] * pair[0][3]
            if (area>=self.min_area*1.25):
                split_left, split_right = self.split_pair(image,cnts[0])
        if (split_left is not None):
            left = self._detect_one_and_size(orig_img=image, image=split_left)
            right = self._detect_one_and_size(orig_img=image, image=split_right)
            if (left is not None and right is not None):
                pair = [left, right]

        pair = sorted(pair, key=lambda x: x[1][0][0], reverse=True)
        return pair if len(pair)==2 else None

    def split_pair(self, img, cnt):
        h,w,_ = img.shape
        hull = cv2.convexHull(cnt,returnPoints = False)
        hull = sorted(hull, reverse=True)
        hull = np.array(hull)
        defects = cv2.convexityDefects(cnt,hull)
        if (defects is None or len(defects)<2):
            return None, None
        defects = sorted(defects, key= lambda x: x[0][3],reverse=True)
        defects = np.array(defects)
        fars = []
        for i in range(2):
            s,e,f,d = defects[i][0]
            far = tuple(cnt[f][0])
            fars.append(far)
        if (fars[0]==fars[1]): 
            fars = np.array(fars)
            fars[0][1]-=1
        fars = sorted(fars, key=lambda x: x[1])
        p1,p2=helper.extend_line(fars[0],fars[1],1000)
        pts_r = np.array([
            0,0, p1[0],p1[1], p2[0],p2[1], 0,h
        ])
        pts_l = np.array([
            w,0, w,h, p2[0],p2[1], p1[0],p1[1]
        ])
        pts_r = pts_r.reshape((-1,1,2))
        pts_l = pts_l.reshape((-1,1,2))
        right = cv2.fillPoly(img.copy(),[pts_r],(0,0,0))
        left = cv2.fillPoly(img.copy(),[pts_l],(0,0,0))
        return left, right