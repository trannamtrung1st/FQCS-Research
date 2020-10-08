from __future__ import print_function
import cv2
import numpy as np
import os
from imutils import perspective
import helper
from scipy.spatial import distance as dist

os.chdir("FQCS_detector")
img1 = cv2.imread("data/1/left/0.jpg")
# img1 = cv2.flip(img1, 1)
img2 = cv2.imread("data/1/left/29.jpg")
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
detector = cv2.SIFT_create(contrastThreshold=0)
kp1, descriptors1 = detector.detectAndCompute(img1, None)
kp2, descriptors2 = detector.detectAndCompute(img2, None)
#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv2.FlannBasedMatcher()
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# Sort them in the order of their distance.
good_matches = sorted(good_matches, key = lambda x:x.distance)
# good_matches = good_matches[:10]

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)  # adding offset

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

# Draw bounding box in Red
img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),1, cv2.LINE_AA)

cv2.imshow("result", img3)
cv2.waitKey()

h,w,_ = img3.shape
rect = cv2.minAreaRect(dst)
box = cv2.boxPoints(rect)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
cv2.drawContours(img3, [box.astype("int")], -1, (0, 255, 0), 2)
cv2.imshow("result", img3)
cv2.waitKey()

(tl, tr, br, bl) = box
(tltrX, tltrY) = helper.midpoint(tl, tr)
(blbrX, blbrY) = helper.midpoint(bl, br)
(tlblX, tlblY) = helper.midpoint(tl, bl)
(trbrX, trbrY) = helper.midpoint(tr, br)
dimA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dimB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

#output
cv2.putText(img3, "{:.1f}px".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 0), 2)
cv2.putText(img3, "{:.1f}px".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 0), 2)

imh, imw, _ = img2.shape
original = img2.copy()
width = int(min(rect[1]))
height = int(max(rect[1]))
tl,tr,br,bl = [0,0],[width-1, 0],[width-1,height-1],[0,height-1]
# dst_pts = np.array([br,bl,tl,tr], dtype="float32")
dst_pts = np.array([tr,tl,bl,br], dtype="float32")
box[:,0]-=img1.shape[1]
M = cv2.getPerspectiveTransform(box, dst_pts)
warped = cv2.warpPerspective(original, M, (width, height))
cv2.imshow("final", warped)
cv2.waitKey()