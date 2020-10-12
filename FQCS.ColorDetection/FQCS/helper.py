import numpy as np
import imutils
import cv2

def change_contrast_and_brightness(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# [get-psnr]
def getPSNR(I1, I2):
    s1 = cv2.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def getMSSISM(i1, i2, C1, C2):
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv2.mean(ssim_map)       # mssim = average of ssim map
    return mssim

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def diff_image(test, true):
    test = test.copy()
    test[test!=0]=255
    diff = test - true
    dist = np.linalg.norm(diff)
    return dist

def extend_line(p1,p2,length):
    len12 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    p3,p4 = np.zeros((2,), dtype="int"),np.zeros((2,), dtype="int")
    p4[0] = p2[0] + (p2[0] - p1[0]) / len12 * length
    p4[1] = p2[1] + (p2[1] - p1[1]) / len12 * length
    p3[0] = p1[0] - (p2[0] - p1[0]) / len12 * length
    p3[1] = p1[1] - (p2[1] - p1[1]) / len12 * length
    return p3,p4

def brightness(hsv):
    return np.mean(hsv[::2])

def get_hist_hsv(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    return hist_h, hist_s, hist_v

def get_hist_bgr(img):
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    return hist_b, hist_g, hist_r

def match_rotation(img, true_img):
    img = cv2.resize(img, (true_img.shape[1], true_img.shape[0]))
    min_deg, min_diff = get_rotation_match(img, true_img)
    if (min_deg is not None):
        img = rotate_image(img, min_deg)
    return img

def get_rotation_match(test, true):
    min_diff = diff_image(test, true)
    min_deg = None
    for deg in np.arange(1, 91, 0.1):
        r_test = rotate_image(test, deg)
        dist = diff_image(r_test, true)
        print(dist)
        if (dist<min_diff):
            min_diff = dist
            min_deg = deg
        else: break

    if (min_deg is None):
        for deg in np.arange(1, 91, 0.1):
            r_test = rotate_image(test, -deg)
            dist = diff_image(r_test, true)
            print(dist)
            if (dist<min_diff):
                min_diff = dist
                min_deg = -deg
            else: break
    
    return min_deg, min_diff

def find_contours_box(c, img):
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

def fill_countours(image, cnts):
    h, w = image.shape[:2]
    mask = np.zeros((h,w), dtype="ubyte")
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    image[mask<127] = 0
    return image
        

def get_warped_box(img, rect, box):
    width = int(min(rect[1]))
    height = int(max(rect[1]))
    tl,tr,br,bl = [0,0],[width-1, 0],[width-1,height-1],[0,height-1]
    # dst_pts = np.array([br,bl,tl,tr], dtype="float32")
    dst_pts = np.array([tr,tl,bl,br], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped
