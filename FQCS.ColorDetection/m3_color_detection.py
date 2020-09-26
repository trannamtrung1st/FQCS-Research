import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

# ori_img = cv2.imread('data/diff/13.jpg')

# def fill(img, h, w):
#     img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
#     return img
        
# def horizontal_shift(img, ratio=0.0):
#     if ratio > 1 or ratio < 0:
#         print('Value should be less than 1 and greater than 0')
#         return img
#     ratio = random.uniform(-ratio, ratio)
#     h, w = img.shape[:2]
#     to_shift = w*ratio
#     if ratio > 0:
#         img = img[:, :int(w-to_shift), :]
#     if ratio < 0:
#         img = img[:, int(-1*to_shift):, :]
#     img = fill(img, h, w)
#     return img

# def vertical_shift(img, ratio=0.0):
#     if ratio > 1 or ratio < 0:
#         print('Value should be less than 1 and greater than 0')
#         return img
#     ratio = random.uniform(-ratio, ratio)
#     h, w = img.shape[:2]
#     to_shift = h*ratio
#     if ratio > 0:
#         img = img[:int(h-to_shift), :, :]
#     if ratio < 0:
#         img = img[int(-1*to_shift):, :, :]
#     img = fill(img, h, w)
#     return img

# def brightness(img, low, high):
#     value = random.uniform(low, high)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv = np.array(hsv, dtype = np.float64)
#     hsv[:,:,1] = hsv[:,:,1]*value
#     hsv[:,:,1][hsv[:,:,1]>255]  = 255
#     hsv[:,:,2] = hsv[:,:,2]*value 
#     hsv[:,:,2][hsv[:,:,2]>255]  = 255
#     hsv = np.array(hsv, dtype = np.uint8)
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return img

# def zoom(img, value):
#     if value > 1 or value < 0:
#         print('Value for zoom should be less than 1 and greater than 0')
#         return img
#     value = random.uniform(value, 1)
#     h, w = img.shape[:2]
#     h_taken = int(value*h)
#     w_taken = int(value*w)
#     h_start = random.randint(0, h-h_taken)
#     w_start = random.randint(0, w-w_taken)
#     img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
#     img = fill(img, h, w)
#     return img

# def channel_shift(img, value):
#     value = int(random.uniform(-value, value))
#     img = img + value
#     img[:,:,:][img[:,:,:]>255]  = 255
#     img[:,:,:][img[:,:,:]<0]  = 0
#     img = img.astype(np.uint8)
#     return img

# def horizontal_flip(img, flag):
#     if flag:
#         return cv2.flip(img, 1)
#     else:
#         return img

# def vertical_flip(img, flag):
#     if flag:
#         return cv2.flip(img, 0)
#     else:
#         return img

# def rotation(img, angle):
#     angle = int(random.uniform(-angle, angle))
#     h, w = img.shape[:2]
#     M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
#     img = cv2.warpAffine(img, M, (w, h))
#     return img

# # def fill_mode(img, left, right):
# #     nearest = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_REPLICATE)
# #     reflect = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_REFLECT)
# #     wrap = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_WRAP)
# #     constant= cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(255, 0, 0))
    
# #     plt.subplot(221),plt.imshow(nearest,'gray'),plt.title('NEAREST'),plt.axis('off')
# #     plt.subplot(222),plt.imshow(reflect,'gray'),plt.title('REFLECT'),plt.axis('off')
# #     plt.subplot(223),plt.imshow(wrap,'gray'),plt.title('WRAP'),plt.axis('off')
# #     plt.subplot(224),plt.imshow(constant,'gray'),plt.title('CONSTANT'),plt.axis('off')
# # def horizontal_shift_mode(img, ratio):
# #     if ratio > 1 or ratio < 0:
# #         print('Value for horizontal shift should be less than 1 and greater than 0')
# #         return img
# #     ratio = random.uniform(-ratio, ratio)
# #     h, w = img.shape[:2]
# #     to_shift = int(w*ratio)
# #     if ratio > 0:
# #         img = img[:, :w-to_shift, :]
# #         fill_mode(img, to_shift, 0)
# #     if ratio < 0:
# #         img = img[:, -1*to_shift:, :]
# #         fill_mode(img, 0, -1*to_shift)

# # cv2.imshow('Result', img)
# # cv2.waitKey(10000000)

# count = 0
# start = 0.1
# for i in range(3):
#     img = horizontal_shift(ori_img, start)
#     start+=0.1
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

# start = 0.1
# for i in range(3):
#     img = vertical_shift(ori_img, start)
#     start+=0.1
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

# start = (0.05, 2)
# for i in range(10):
#     img = brightness(ori_img, start[0], start[1])
#     start=(start[0]+0.1,start[1]+0.1)
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

# start = 0.05
# for i in range(10):
#     img = zoom(ori_img, start)
#     start+=0.1
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

# start = 10
# for i in range(10):
#     img = channel_shift(ori_img, start)
#     start+=10
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

# img = horizontal_flip(ori_img, True)
# cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
# img = vertical_flip(ori_img, True)
# cv2.imwrite("data/diff/13"+str(count)+".jpg", img)

# start = 10
# for i in range(36):
#     img = rotation(ori_img, start)    
#     start+=10
#     cv2.imwrite("data/diff/13"+str(count)+".jpg", img)
#     count+=1

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def preprocess(img):
#     return (cv2.blur(img,(10,10)))

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     preprocessing_function=preprocess
# )

# target_size = (256, 256)
# generator = datagen.flow_from_directory(
#     "data",
#     target_size=target_size,
#     batch_size=16,
#     class_mode="categorical"
# )

# inp_shape = (256, 256, 3)
# model = keras.Sequential([
#     L.Input(inp_shape),
#     L.Conv2D(filters=5, kernel_size=(3,3), activation="relu"),
#     L.MaxPooling2D(pool_size=(2, 2)),
#     L.Flatten(),
#     L.Dense(units=2, activation="softmax")
# ])
# model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
# model.summary()
# model.fit(generator, epochs=50)
# model.save("saved_color_detection_model")

model = keras.models.load_model("saved_color_detection_model")
model.compile()
model.summary()
img = cv2.imread("test.jpg")
img = cv2.resize(img, (256, 256))
inp = np.array([
    img
])
pred = model.predict(inp)
print(pred)