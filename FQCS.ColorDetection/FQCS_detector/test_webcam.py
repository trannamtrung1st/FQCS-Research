import cv2

cap = cv2.VideoCapture(1)

while (True):
    _,img = cap.read()
    cv2.imshow("Cap", img)
    if (cv2.waitKey(30) == ord('q')):
        break