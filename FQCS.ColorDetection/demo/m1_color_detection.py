# Python code for Multiple Color Detection 
  
  
import numpy as np 
import cv2 
import itertools

edge = 500
all_color_range = np.array([
    # ([0, 0, 255*0.4],[360, 255*0.2, 255]),#white
    # ([0, 0, 255*0],[360, 255, 255*0.1]),#black
    # ([0, 255*0.6, 255*0.6],[5, 255, 255]),#red
    # ([175, 255*0.6, 255*0.4],[180, 255, 255]),#red
    # ([100, 255*0.2, 255*0.2],[125, 255, 255]),#blue
    # ([40, 255*0.4, 255*0.2],[80, 255, 255]),#green
    # ([105, 255*0.1, 255*0.1],[115, 255*0.2, 255*0.9]),#dark_blue
    # ([105, 255*0.2, 255*0.1],[115, 255*0.3, 255*0.9]),#dark_blue
    # ([105, 255*0.3, 255*0.1],[115, 255*0.4, 255*0.9]),#dark_blue
    # ([105, 255*0.4, 255*0.1],[115, 255*0.5, 255*0.9]),#dark_blue
    # ([105, 255*0.5, 255*0.1],[115, 255*0.6, 255*0.9]),#dark_blue
    # ([105, 255*0.6, 255*0.1],[115, 255*0.7, 255*0.9]),#dark_blue
    # ([105, 255*0.7, 255*0.1],[115, 255*0.8, 255*0.9]),#dark_blue
    # ([105, 255*0.8, 255*0.1],[115, 255*0.9, 255*0.9]),#dark_blue
    # ([105, 255*0.9, 255*0.1],[115, 255*1, 255*0.9]),#dark_blue
    ([15, 255*0.1, 255*0.1],[25, 255*0.2, 255*0.9]),#yellow
    ([15, 255*0.2, 255*0.1],[25, 255*0.3, 255*0.9]),#yellow
    ([15, 255*0.3, 255*0.1],[25, 255*0.4, 255*0.9]),#yellow
    ([15, 255*0.4, 255*0.1],[25, 255*0.5, 255*0.9]),#yellow
    ([15, 255*0.5, 255*0.1],[25, 255*0.6, 255*0.9]),#yellow
    ([15, 255*0.6, 255*0.1],[25, 255*0.7, 255*0.9]),#yellow
    ([15, 255*0.7, 255*0.1],[25, 255*0.8, 255*0.9]),#yellow
    ([15, 255*0.8, 255*0.1],[25, 255*0.9, 255*0.9]),#yellow
    ([15, 255*0.9, 255*0.1],[25, 255*1, 255*0.9]),#yellow
])


for i in range(len(all_color_range)):
    color_range = np.array([all_color_range[i]])

    # Reading the video from the 
    # webcam in image frames 
    imageFrame = cv2.imread("test-diff-2.jpg")
    imageFrame = cv2.resize(imageFrame, (edge, edge))
    imageFrame = cv2.blur(imageFrame, (10, 10))
    # imageFrame = cv2.fastNlMeansDenoisingColoredMulti(imageFrame,None,10,10,7,21)

    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
    masks = [(color_range[i][0],cv2.inRange(hsvFrame, color_range[i][0],color_range[i][1])) for i in range(color_range.shape[0])]
        
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 
        
    # For red color 
    masks = [(masks[i][0], cv2.dilate(masks[i][1], kernel)) for i in range(len(masks))]
    # res_masks = [cv2.bitwise_and(imageFrame, imageFrame,  
    #                             mask = masks[i]) for i in range(len(masks))]

    # Creating contour to track red color 
    c_h = [(masks[i][0],cv2.findContours(masks[i][1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)) for i in range(len(masks))]

    for i in range(len(c_h)):
        contours = c_h[i][1][0]
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > edge*edge*0.02**2): 
                x, y, w, h = cv2.boundingRect(contour) 
                color = c_h[i][0]
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                # cv2.putText(imageFrame, str(color) +","+ str(area), (x, y), 
                #             cv2.FONT_HERSHEY_SIMPLEX,  
                #             0.5, (0, 0, 0)) 


    cv2.namedWindow("Test")
    cv2.imshow("Test",imageFrame)
    cv2.waitKey(100000)