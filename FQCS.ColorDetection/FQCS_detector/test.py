import cv2
import numpy as np
import os

os.chdir("FQCS_detector")
img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]

hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
defects = sorted(defects, key= lambda x: x[0][3],reverse=True)
defects = np.array(defects)

for cur_def in defects[:2]:
    s,e,f,d = cur_def[0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    print(d)
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()