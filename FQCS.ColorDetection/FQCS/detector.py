from config_model import DetectorConfig
import helper
import json
import numpy as np

class FQCSDetector():
    
    def _find_cnt_using_edge(self, image):
        return
    def _find_cnt_using_thresh(self, image):
        return
    def _find_cnt_using_range(self, image):
        return

    def __init__(self, config: DetectorConfig = None):
        self.config = config if config is not None else DetectorConfig()
        self.find_cnt_funcs = {
            "edge": self._find_cnt_using_edge,
            "thresh": self._find_cnt_using_thresh,
            "range": self._find_cnt_using_range,
        }
        
    def save_config(self, path):
        with open(path, 'w') as out:
            config_dict = self.config.get_dict()
            json.dump(config_dict, out, indent=2)
            
    def load_config(self, path):
        with open(path) as inp:
            data = json.load(inp)
            self.config = DetectorConfig(**data)
    
    def detect_one_and_size(self, orig_img: np.ndarray, image: np.ndarray):
        # start
        h, w = image.shape[:2]
        find_contours = self.find_cnt_funcs[self.config.find_contours_func]
        cnts = find_contours(image)
        helper.fill_contours(image, cnts)
        c = cnts[0]
        rect,dimA,dimB,box,tl,tr,br,bl = helper.find_cnt_box(c, image)
        warped = helper.get_warped_box(image, rect, box)
        return (warped,box,dimA,dimB)

    def detect_pair_and_size(self, image: np.ndarray):
        # start
        pair = []
        h, w = image.shape[:2]
        find_contours = self.find_cnt_funcs[self.config.find_contours_func]
        cnts = find_contours(image)
        helper.fill_contours(image, cnts)
        min_x,max_x = w,0
        dr = self.config.detect_range
        from_x, to_x = w*dr[0],w*dr[1]
        for c in cnts[:2]:
            if cv2.contourArea(c) < self.config.min_area:
                break
            rect,dimA,dimB,box,tl,tr,br,bl = helper.find_cnt_box(c, image)
            cur_min_x = min(tl[0], tr[0], br[0], bl[0])
            cur_max_x = max(tl[0], tr[0], br[0], bl[0])
            min_x = min(cur_min_x, min_x)
            max_x = max(cur_max_x, max_x)
            if (min_x<from_x or max_x>to_x):
                break
            center_val = w-max_x-min_x
            is_center = True if (center_val<=stop_condition) else False
            if (is_center):
                warped = helper.get_warped_box(image, rect, box)
                pair.append((warped,box,dimA,dimB))
        
        split_left, split_right = None, None
        if (self.config.min_area is not None and len(pair)==1):
            area = pair[0][2] * pair[0][3]
            if (area>=self.config.min_area*1.25):
                split_left, split_right = self.split_pair(image,cnts[0])
        if (split_left is not None):
            left = self.detect_one_and_size(orig_img=image, image=split_left)
            right = self.detect_one_and_size(orig_img=image, image=split_right)
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
