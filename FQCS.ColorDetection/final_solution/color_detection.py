import numpy as np 
import cv2 
import matplotlib.pyplot as plt

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
    # END INITS
    # PRELIMINARY COMPUTING
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

def preprocess(img, img_size = (32, 64),blur_val = 0.05, alpha=1, beta=-150, sat_adj=2):
    img = change_contrast_and_brightness(img, alpha, beta)
    img = cv2.blur(img, (round(img.shape[0] * blur_val),round(img.shape[1] * blur_val)))
    img = cv2.resize(img, img_size)
    if (sat_adj!=1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,1]*=sat_adj
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def find_diff(test, true, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue,min_similarity):
    results = np.zeros(matrix)
    for v in range(matrix[0]):
        for h in range(matrix[1]):
            sub_test = test[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]
            sub_true = true[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]

            # hist_diff(sub_left, sub_right, hist_color_thresh)

            psnrv = getPSNR(sub_test, sub_true)
            print("{}dB".format(round(psnrv, 3)))
            mssimv = None
            if (psnrv < psnrTriggerValue and psnrv):
                mssimv = getMSSISM(sub_test, sub_true, C1, C2)
                # bias
                mssimv = np.array(mssimv)/biases[v,h]
                print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[0] * 100, 2), round(mssimv[1] * 100, 2),
                                                        round(mssimv[2] * 100, 2)))

            mssimv = None if mssimv is None else mssimv[:3]
            has_diff = 1 if mssimv is not None and mssimv[mssimv<min_similarity].any() else 0
            results[v,h] = has_diff
            
            # fig,axs = plt.subplots(1, 2)
            # if has_diff:
            #     plt.title("Different")
            # axs[0].imshow(sub_test)
            # axs[1].imshow(sub_true)
            # plt.show()   

    has_diff = results[results==1].any()
    return results, has_diff

def detect_color_difference(left, right, true_left, true_right, 
    biases = None, C1=6.5025,
    C2=58.5225,psnrTriggerValue = 40,
    matrix = (4, 4),min_similarity = 0.8):
    # START
    if biases is None:
        biases = np.ones(matrix)
    # must be divisible 
    ver_step = left.shape[0]//matrix[0]
    hor_step = left.shape[1]//matrix[1]

    left_results, left_has_diff = find_diff(left, true_left, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue, min_similarity)
    right_results, right_has_diff = find_diff(right, true_right, matrix, ver_step, hor_step, biases,C1,C2,psnrTriggerValue,min_similarity)
    return left_results, left_has_diff, right_results, right_has_diff   
