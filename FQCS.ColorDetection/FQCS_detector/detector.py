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

    def find_color_diff(self, test, true, amp_thresh, amplify_rate,max_diff):
        test_hist = helper.get_hist_bgr(test)
        true_hist = helper.get_hist_bgr(true)
        list_dist = np.zeros((3,))
        w,h,_= true.shape

        max_dist = w*h
        amp_thresh = max_dist*amp_thresh
        print("Amp",amp_thresh)
        # output
        for i in range(3):
            dist = np.linalg.norm(test_hist[i]-true_hist[i])
            list_dist[i] = dist
        mean = np.mean(list_dist)
        if (mean>amp_thresh):
            list_dist*=(mean/amp_thresh)**amplify_rate
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
        amp_thresh = None,amplify_rate=None, max_diff = None):
        # START
        left_results, left_has_diff = self.find_color_diff(left, true_left,amp_thresh, amplify_rate,max_diff)
        right_results, right_has_diff = self.find_color_diff(right, true_right,amp_thresh, amplify_rate, max_diff)
        return left_results, left_has_diff, right_results, right_has_diff

    def detect_pair_and_size(self, image: np.ndarray, alpha = 1.0,
        beta = 0,canny_threshold1 = 40,canny_threshold2 = 100,
        kernel = (5, 5), sigma_x=0, 
        color_threshold=0, color_max_val=255, min_area=400,stop_condition=0):
        # start
        pair = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, kernel, 0)
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        edged = cv2.Canny(enhanced, canny_threshold1, canny_threshold2)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        h, w = image.shape[:2]

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)

        orig = image.copy()
        min_x,max_x = w,0
        for c in cnts[:2]:
            if cv2.contourArea(c) < min_area:
                break
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            # output
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = helper.midpoint(tl, tr)
            (blbrX, blbrY) = helper.midpoint(bl, br)
            (tlblX, tlblY) = helper.midpoint(tl, bl)
            (trbrX, trbrY) = helper.midpoint(tr, br)
            cur_min_x = min(tl[0], tr[0], br[0], bl[0])
            cur_max_x = max(tl[0], tr[0], br[0], bl[0])
            min_x = min(cur_min_x, min_x)
            max_x = max(cur_max_x, max_x)
            center_val = w-max_x-min_x
            is_center = True if (center_val<=stop_condition) else False
            dimA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dimB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            #output
            cv2.putText(orig, "{:.1f}px".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            cv2.putText(orig, "{:.1f}px".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            
            imh, imw, _ = image.shape
            if (is_center):
                original = image.copy()
                width = int(min(rect[1]))
                height = int(max(rect[1]))
                tl,tr,br,bl = [0,0],[width-1, 0],[width-1,height-1],[0,height-1]
                # dst_pts = np.array([br,bl,tl,tr], dtype="float32")
                dst_pts = np.array([tr,tl,bl,br], dtype="float32")
                M = cv2.getPerspectiveTransform(box, dst_pts)
                warped = cv2.warpPerspective(original, M, (width, height))
                pair.append((warped,box,dimA,dimB))
            
        # output
        cv2.imshow('detect', orig)
        cv2.waitKey(1)

        return pair if (len(pair) == 2) else None