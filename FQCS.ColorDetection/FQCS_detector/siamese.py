import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.utils import plot_model
import os
import cv2

def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    return img

os.chdir("FQCS_detector")
# base_dir = r'data/1'
# train_test_split = 0.7
# no_of_files_in_each_class = 10

# #Read all the folders in the directory
# folder_list = os.listdir(base_dir)
# print( len(folder_list), "categories found in the dataset")

# #Declare training array
# cat_list = []
# x = []
# y = []
# y_label = 0

# #Using just 5 images per category
# for folder_name in folder_list:
#     files_list = os.listdir(os.path.join(base_dir, folder_name))
#     temp=[]
#     for file_name in files_list[:no_of_files_in_each_class]:
#         temp.append(len(x))
#         img = cv2.imread(os.path.join(base_dir, folder_name, file_name))
#         img = process(img)
#         x.append(np.asarray(img))
#         y.append(y_label)
#     y_label+=1
#     cat_list.append(temp)

# cat_list = np.asarray(cat_list)
# x = np.asarray(x)/255.0
# y = np.asarray(y)
# print('X, Y shape',x.shape, y.shape, cat_list.shape)       

# train_size = int(len(folder_list)*train_test_split)
# test_size = len(folder_list) - train_size
# print(train_size, 'classes for training and', test_size, ' classes for testing')

# train_files = train_size * no_of_files_in_each_class

# #Training Split
# x_train = x[:train_files]
# y_train = y[:train_files]
# cat_train = cat_list[:train_size]

# #Validation Split
# x_val = x[train_files:]
# y_val = y[train_files:]
# cat_test = cat_list[train_size:]

# print('X&Y shape of training data :',x_train.shape, 'and', y_train.shape, cat_train.shape)
# print('X&Y shape of testing data :' , x_val.shape, 'and', y_val.shape, cat_test.shape)

# def get_batch(batch_size=64):
    
#     temp_x = x_train
#     temp_cat_list = cat_train
#     start=0
#     end=train_size
#     batch_x=[]
        
#     batch_y = np.zeros(batch_size)
#     batch_y[int(batch_size/2):] = 1
#     np.random.shuffle(batch_y)
    
#     class_list = np.random.randint(start, end, batch_size) 
#     batch_x.append(np.zeros((batch_size, 100, 100, 3)))
#     batch_x.append(np.zeros((batch_size, 100, 100, 3)))

#     for i in range(0, batch_size):
#         batch_x[0][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]  
#         #If train_y has 0 pick from the same class, else pick from any other class
#         if batch_y[i]==0:
#             batch_x[1][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]

#         else:
#             temp_list = np.append(temp_cat_list[:class_list[i]].flatten(), temp_cat_list[class_list[i]+1:].flatten())
#             batch_x[1][i] = temp_x[np.random.choice(temp_list)]
            
#     return(batch_x, batch_y)

# #Building a sequential model
# input_shape=(100, 100, 3)
# left_input = Input(input_shape)
# right_input = Input(input_shape)

# W_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 1e-2)
# b_init = keras.initializers.RandomNormal(mean = 0.5, stddev = 1e-2)

# model = keras.models.Sequential([
#     keras.layers.Conv2D(64, (10,10), activation='relu', input_shape=input_shape, kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(128, (7,7), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
#     keras.layers.MaxPooling2D(2,2),
#     keras.layers.Conv2D(128, (4,4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
#     keras.layers.MaxPooling2D(2,2),
#     keras.layers.Conv2D(256, (4,4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
#     keras.layers.MaxPooling2D(2,2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=W_init, bias_initializer=b_init)
# ])

# encoded_l = model(left_input)
# encoded_r = model(right_input)

# subtracted = keras.layers.Subtract()([encoded_l, encoded_r])
# prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(subtracted)
# siamese_net = Model([left_input, right_input], prediction)

# optimizer= Adam(learning_rate=0.0001)
# siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)

# # plot_model(siamese_net, show_shapes=True, show_layer_names=True)

# def nway_one_shot(model, n_way, n_val):
    
#     temp_x = x_val
#     temp_cat_list = cat_test
#     batch_x=[]
#     x_0_choice=[]
#     n_correct = 0
   
#     class_list = np.random.randint(0, len(folder_list)-1, n_val)

#     for i in class_list:  
#         j = np.random.choice(cat_list[i])
#         temp=[]
#         temp.append(np.zeros((n_way, 100, 100, 3)))
#         temp.append(np.zeros((n_way, 100, 100, 3)))
#         for k in range(0, n_way):
#             temp[0][k] = x[j]
            
#             if k==0:
#                 #print(i, k, j, np.random.choice(cat_list[i]))
#                 temp[1][k] = x[np.random.choice(cat_list[i])]
#             else:
#                 #print(i, k, j, np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten())))
#                 temp[1][k] = x[np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten()))]

#         result = siamese_net.predict(temp)
#         result = result.flatten().tolist()
#         result_index = result.index(min(result))
#         if result_index == 0:
#             n_correct = n_correct + 1
#     print(n_correct, "correctly classified among", n_val)
#     accuracy = (n_correct*100)/n_val
#     return accuracy

# epochs = 100
# n_way = 4
# n_val = 100
# batch_size = 64

# loss_list=[]
# accuracy_list=[]
# for epoch in range(1,epochs):
#     batch_x, batch_y = get_batch(batch_size)
#     loss = siamese_net.train_on_batch(batch_x, batch_y)
#     loss_list.append((epoch,loss))
#     print('Epoch:', epoch, ', Loss:',loss)
#     if epoch%10 == 0:
#         siamese_net.save("saved_model")
#         print("=============================================")
#         accuracy = nway_one_shot(model, n_way, n_val)
#         accuracy_list.append((epoch, accuracy))
#         print('Accuracy as of', epoch, 'epochs:', accuracy)
#         print("=============================================")
#         if(accuracy>99):
#             print("Achieved more than 90% Accuracy")
#             #break

siamese_net = keras.models.load_model("saved_model")
siamese_net.compile()
img1 = cv2.imread("true_left.jpg")
img2 = cv2.imread("true_left_dirty.jpg")

img1 = process(img1)
img2 = process(img2)

X = [
    np.array([img1]),
    np.array([img2]),
]
y = siamese_net.predict(X)
print(y)