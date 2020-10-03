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
        img = cv2.blur(img, (round(img.shape[0] * blur_val),round(img.shape[1] * blur_val)))
        img = cv2.resize(img, img_size)
        return img

    def find_color_diff(self, test, true, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue,min_similarity):
        results = np.ones((matrix[0], matrix[1], 3))
        for v in range(matrix[0]):
            for h in range(matrix[1]):
                sub_test = test[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]
                sub_true = true[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]
                psnrv = helper.getPSNR(sub_test, sub_true)
                print("{}dB".format(round(psnrv, 3)))
                mssimv = None
                if (psnrv < psnrTriggerValue and psnrv):
                    mssimv = helper.getMSSISM(sub_test, sub_true, C1, C2)
                    # bias
                    mssimv = np.array(mssimv)/biases[v,h]
                    print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[0] * 100, 2), round(mssimv[1] * 100, 2),
                                                            round(mssimv[2] * 100, 2)))

                mssimv = None if mssimv is None else mssimv[:3]
                has_diff = 1 if mssimv is not None and mssimv[mssimv<min_similarity].any() else 0
                if mssimv is not None:
                    results[v,h] = mssimv

                test_hist = helper.get_hist_bgr(sub_test)
                true_hist = helper.get_hist_bgr(sub_true)
                list_dist = np.zeros((3,))
                # output
                for i in range(3):
                    # fig,axs = plt.subplots(1, 2)
                    # axs[0].plot(test_hist[i])
                    # axs[1].plot(true_hist[i])
                    # plt.show()
                    dist = np.linalg.norm(test_hist[i]-true_hist[i])
                    print("Dist", dist)
                    list_dist[i] = dist
                max_dist = np.max(list_dist)
                print("Max", max_dist)

                # output
                fig,axs = plt.subplots(1, 2)
                if has_diff:
                    plt.title("Different")
                axs[0].imshow(sub_test)
                axs[1].imshow(sub_true)
                plt.show()   

        ssim_has_diff = results[results<min_similarity].any()
        return results, ssim_has_diff

    def detect_color_difference(self, left, right, true_left, true_right, 
        biases = None, C1=6.5025,
        C2=58.5225,psnrTriggerValue = 40,
        matrix = (4, 4),min_similarity = 0.6):
        # START
        if biases is None:
            biases = np.ones(matrix)
        # must be divisible 
        ver_step = left.shape[0]//matrix[0]
        hor_step = left.shape[1]//matrix[1]

        left_results, left_has_diff = self.find_color_diff(left, true_left, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue, min_similarity)
        right_results, right_has_diff = self.find_color_diff(right, true_right, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue,min_similarity)
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

        # th, im_th = cv2.threshold(edged, color_threshold, color_max_val, cv2.THRESH_BINARY)
        # im_floodfill = im_th.copy()
        # mask = np.zeros((h+2, w+2), np.uint8)
        # cv2.floodFill(im_floodfill, mask, (0,0), 255)
        # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # im_out = im_th | im_floodfill_inv

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)

        # image[im_out < 200] = 0
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