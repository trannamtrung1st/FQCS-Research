import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def change_contrast_and_brightness(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    # alpha = 1.0 # Simple contrast control
    # beta = 0    # Simple brightness control
    # Initialize values
    # print(' Basic Linear Transforms ')
    # print('-------------------------')
    # try:
    #     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #     beta = int(input('* Enter the beta value [0-100]: '))
    # except ValueError:
    #     print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
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

# [get-psnr]
# [get-mssim]
def getMSSISM(i1, i2):
    # # increase mean decrease sensitive ... (manual test)
    # # C1 = 6.5025
    # # C2 = 58.5225
    # C1 = 30
    # C2 = 100
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

def hist_diff(i1, i2, thresh):
    hist_left_r = cv2.calcHist([i1],[0],None,[256],[0,256])
    hist_left_g = cv2.calcHist([i1],[1],None,[256],[0,256])
    hist_left_b = cv2.calcHist([i1],[2],None,[256],[0,256])
    hist_right_r = cv2.calcHist([i2],[0],None,[256],[0,256])
    hist_right_g = cv2.calcHist([i2],[1],None,[256],[0,256])
    hist_right_b = cv2.calcHist([i2],[2],None,[256],[0,256])
    
    dist_r = hist_left_r-hist_right_r
    dist_g = hist_left_g-hist_right_g
    dist_b = hist_left_b-hist_right_b
    print("Dist:",dist_r[dist_r>thresh].any(),dist_g[dist_g>thresh].any(),dist_b[dist_b>thresh].any(), thresh)

# -------------- PARAM ---------------------
# increase mean decrease sensitive ... (manual test)
# C1 = 6.5025
# C2 = 58.5225
C1 = 6.5025/3
C2 = 58.5225/3

psnrTriggerValue = 40
img_size = (32, 64)
blur_val = 0.05
left = cv2.imread("d_left_3.jpg")
left = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
right = cv2.imread("d_right_3.jpg")
right = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
left[:,:,1]*=3
right[:,:,1]*=3

left = cv2.cvtColor(left, cv2.COLOR_HSV2RGB)
right = cv2.cvtColor(right, cv2.COLOR_HSV2RGB)

fig,axs = plt.subplots(1, 2)
axs[0].imshow(left)
axs[1].imshow(right)
plt.show()



right = change_contrast_and_brightness(right, 1, -150)
left = change_contrast_and_brightness(left, 1, -150)

# SEGMENT MATRIX
# matrix = (2, 2)
matrix = (4, 4)
# matrix = (8, 8)

# BIASES MATRIX
biases = np.array([
    [0.9, 1.1, 1, 0.8],
    [1, 0.9, 0.9, 0.8],
    [0.65, 0.65, 0.65, 0.65],
    [0.1, 1.1, 1.1, 0.1],
])

min_similarity = 0.8
# -------------------------------------------

fig,axs = plt.subplots(1, 2)
axs[0].imshow(left)
axs[1].imshow(right)
plt.show()

left = cv2.blur(left, (round(left.shape[0] * blur_val),round(left.shape[1] * blur_val)))
right = cv2.blur(right, (round(right.shape[0] * blur_val),round(right.shape[1] * blur_val)))
left = cv2.resize(left, img_size)
right = cv2.resize(right, img_size)

psnrv = getPSNR(left, right)
print("{}dB".format(round(psnrv, 3)))
if (psnrv < psnrTriggerValue and psnrv):
    mssimv = getMSSISM(left, right)
    print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[0] * 100, 2), round(mssimv[1] * 100, 2),
                                            round(mssimv[2] * 100, 2)))

fig,axs = plt.subplots(1, 2)
axs[0].imshow(left)
axs[1].imshow(right)
plt.show()

# must be divisible 
ver_step = left.shape[0]//matrix[0]
hor_step = left.shape[1]//matrix[1]
hist_color_thresh = (ver_step * hor_step * 0.1)

for v in range(matrix[0]):
    for h in range(matrix[1]):
        sub_left = left[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]
        sub_right = right[v*ver_step:(v+1)*ver_step, h*hor_step: (h+1)*hor_step]

        # hist_diff(sub_left, sub_right, hist_color_thresh)

        psnrv = getPSNR(sub_left, sub_right)
        print("{}dB".format(round(psnrv, 3)))
        mssimv = None
        if (psnrv < psnrTriggerValue and psnrv):
            mssimv = getMSSISM(sub_left, sub_right)
            # bias
            mssimv = np.array(mssimv)/biases[v,h]
            print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[0] * 100, 2), round(mssimv[1] * 100, 2),
                                                    round(mssimv[2] * 100, 2)))

        fig,axs = plt.subplots(1, 2)
        mssimv = None if mssimv is None else mssimv[:3]
        if mssimv is not None and mssimv[mssimv<min_similarity].any():
            plt.title("Different")
        axs[0].imshow(sub_left)
        axs[1].imshow(sub_right)
        plt.show()        
