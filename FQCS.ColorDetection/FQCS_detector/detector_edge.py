from scipy.spatial import distance as dist
import numpy as np
import cv2
import matplotlib.pyplot as plt
import helper
import imutils
from imutils import perspective

class FQCSDetector:

    def preprocess_for_cd(self, img, img_size = (32, 64),blur_val = 0.05, alpha=1, beta=-150, sat_adj=2):
        if (sat_adj!=1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:,:,1]*=sat_adj
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = helper.change_contrast_and_brightness(img, alpha, beta)
        if blur_val is not None:
            img = cv2.blur(img, (round(img.shape[0] * blur_val),round(img.shape[1] * blur_val)))
        img = cv2.resize(img, img_size)
        return img

    def find_color_diff(self, test, true, amp_thresh,supp_thresh, amplify_rate,max_diff):
        test_hist = helper.get_hist_bgr(test)
        true_hist = helper.get_hist_bgr(true)
        list_dist = np.zeros((3,))
        w,h,_= true.shape

        max_dist = w*h
        print("Amp",amp_thresh)
        # output
        for i in range(3):
            diff = np.abs(test_hist[i]-true_hist[i])
            diff[diff<supp_thresh]=0
            dist = np.linalg.norm(diff)
            if (dist>amp_thresh[i]):
                dist*=(dist/amp_thresh[i])**amplify_rate
            list_dist[i] = dist
        print(list_dist)
        sum_dist = np.sum(list_dist)
        print(sum_dist, max_dist)
        avg = sum_dist/max_dist
        print("Avg(%):", avg)

        # output
        fig,axs = plt.subplots(1, 2)
        axs[0].imshow(test)
        axs[1].imshow(true)
        plt.show()   

        return sum_dist, avg>=max_diff

    def detect_color_difference(self, left, right, true_left, true_right,
        amp_thresh = None,supp_thresh=None,amplify_rate=None, max_diff = None):
        # START
        left_results, left_has_diff = self.find_color_diff(left, true_left,amp_thresh,supp_thresh, amplify_rate,max_diff)
        right_results, right_has_diff = self.find_color_diff(right, true_right,amp_thresh, supp_thresh,amplify_rate, max_diff)
        return left_results, left_has_diff, right_results, right_has_diff

    def find_contours(self, image, kernel,d_kernel,e_kernel,alpha,beta,canny_threshold1, canny_threshold2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, kernel, 0)
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        edged = cv2.Canny(enhanced, canny_threshold1, canny_threshold2)
        edged = cv2.dilate(edged, d_kernel, iterations=1)
        edged = cv2.erode(edged, e_kernel, iterations=1)

        # output
        cv2.imshow("Edged", edged)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        return cnts

    def find_cnt_box(self, c, img):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = helper.midpoint(tl, tr)
        (blbrX, blbrY) = helper.midpoint(bl, br)
        (tlblX, tlblY) = helper.midpoint(tl, bl)
        (trbrX, trbrY) = helper.midpoint(tr, br)
        
        dimA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dimB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        return rect,dimA,dimB,box,tl,tr,br,bl

    def get_warped_cnt(self, img, rect, box):
        width = int(min(rect[1]))
        height = int(max(rect[1]))
        tl,tr,br,bl = [0,0],[width-1, 0],[width-1,height-1],[0,height-1]
        # dst_pts = np.array([br,bl,tl,tr], dtype="float32")
        dst_pts = np.array([tr,tl,bl,br], dtype="float32")
        M = cv2.getPerspectiveTransform(box, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        return warped

    def detect_one_and_size(self, orig_img: np.ndarray, image: np.ndarray, bg_thresh=100, alpha = 1.0,
        beta = 0,canny_threshold1 = 40,canny_threshold2 = 100,
        kernel = (5, 5), d_kernel=None,e_kernel=None,sample_area=None):
        # start
        h, w = image.shape[:2]
        cnts = self.find_contours(image, kernel, d_kernel, e_kernel, alpha, beta, canny_threshold1, canny_threshold2)
        mask = np.zeros((h,w), dtype="ubyte")
        cv2.fillPoly(mask, cnts, (255, 255, 255))
        image[mask<127] = 0
        c = cnts[0]
        rect,dimA,dimB,box,tl,tr,br,bl = self.find_cnt_box(c, image)
        warped = self.get_warped_cnt(image, rect, box)
        return (warped,box,dimA,dimB)

    def detect_pair_and_size(self, image: np.ndarray,bg_thresh=100, alpha = 1.0,
        beta = 0,canny_threshold1 = 40,canny_threshold2 = 100,
        kernel = (5, 5), d_kernel=None,e_kernel=None,sample_area=None,stop_condition=0,detect_range=(0.2,0.8)):
        # start
        pair = []
        h, w = image.shape[:2]
        cnts = self.find_contours(image, kernel, d_kernel, e_kernel, alpha, beta, canny_threshold1, canny_threshold2)
        mask = np.zeros((h,w), dtype="ubyte")
        cv2.fillPoly(mask, cnts, (255, 255, 255))
        image[mask<127] = 0
        orig = image.copy()
        min_x,max_x = w,0
        from_x, to_x = w*detect_range[0],w*detect_range[1]
        min_area = sample_area*0.25 if sample_area is not None else 400
        for c in cnts[:2]:
            if cv2.contourArea(c) < min_area:
                break
            rect,dimA,dimB,box,tl,tr,br,bl = self.find_cnt_box(c, orig)
            cur_min_x = min(tl[0], tr[0], br[0], bl[0])
            cur_max_x = max(tl[0], tr[0], br[0], bl[0])
            min_x = min(cur_min_x, min_x)
            max_x = max(cur_max_x, max_x)
            if (min_x<from_x or max_x>to_x):
                break

            # output
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            cv2.putText(orig, "{:.1f}px".format(dimA),
                        (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            cv2.putText(orig, "{:.1f}px".format(dimB),
                        (br[0], br[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)

            center_val = w-max_x-min_x
            is_center = True if (center_val<=stop_condition) else False
            if (is_center):
                original = image.copy()
                warped = self.get_warped_cnt(original, rect, box)
                pair.append((warped,box,dimA,dimB))
        
        pair = sorted(pair, key=lambda x: x[1][0][0], reverse=True)

        # output
        cv2.imshow('detect', orig)
        cv2.waitKey(1)
        
        split_left, split_right = None, None
        if (sample_area is not None and (len(pair)==2 or len(pair)==1)):
            area = pair[0][2] * pair[0][3]
            if (area>=sample_area*1.25):
                split_left, split_right = self.split_pair(image,cnts[0])
        if (split_left is not None):
            left = self.detect_one_and_size(orig_img=image, image=split_left,bg_thresh=bg_thresh,
                alpha=alpha,beta=beta,canny_threshold1=canny_threshold1,canny_threshold2=canny_threshold2,
                kernel=kernel,sample_area=sample_area)
            cv2.imshow("Splitted", left[0])
            cv2.waitKey()
            right = self.detect_one_and_size(orig_img=image, image=split_right,bg_thresh=bg_thresh,
                alpha=alpha,beta=beta,canny_threshold1=canny_threshold1,canny_threshold2=canny_threshold2,
                kernel=kernel,sample_area=sample_area)
            cv2.imshow("Splitted", right[0])
            cv2.waitKey()
            
            if (left is not None and right is not None):
                pair = [left, right]
                pair = sorted(pair, key=lambda x: x[1][0][0], reverse=True)

        return pair if len(pair)==2 else None

    def split_pair(self, img,cnt):
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
            print(far)
            # cv2.circle(img,far,5,[0,0,255],-1)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            fars.append(far)
        if (fars[0]==fars[1]): 
            fars = np.array(fars)
            fars[0][1]-=1
        fars = sorted(fars, key=lambda x: x[1])

        p1,p2=helper.extend_line(fars[0],fars[1],1000)
        pts_r = np.array([
            0,0, p1[0],p1[1], p2[0],p2[1], 0,h
        ])
        pts_r = pts_r.reshape((-1,1,2))
        pts_l = np.array([
            w,0, w,h, p2[0],p2[1], p1[0],p1[1]
        ])
        pts_l = pts_l.reshape((-1,1,2))
        
        right = cv2.fillPoly(img.copy(),[pts_r],(0,0,0))
        cv2.imshow("Splitted", right)
        cv2.waitKey()
        left = cv2.fillPoly(img.copy(),[pts_l],(0,0,0))
        cv2.imshow("Splitted", left)
        cv2.waitKey()
        return left, right