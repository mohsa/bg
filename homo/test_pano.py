import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import os

from deep_homography_estimation_dg import deep_homography_test
from test_datagen import generate_data

from homography_generator import HomographyGenerator
from constants import *
import datetime
now = datetime.datetime.now

img_rows, img_cols = 180, 320
img_cols_ds = 448

image_folder='../Data/Images/frames_360p'
output_folder = '../output'
weights_path = os.path.join(output_folder, 'weights.h5')

img = cv2.imread(os.path.join(image_folder, 'frames_003333.png'))
img2 = cv2.imread(os.path.join(image_folder, 'frames_003336.png'))

r = img_cols_ds/ img.shape[1]
dim = (img_cols_ds, int(img.shape[0] * r))
img = cv2.cvtColor(cv2.resize(img, (dim), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.resize(img2, (dim), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

im2d = np.zeros((img_rows,img_cols,2), np.uint8)
img1, _, _ = generate_data(img,img_rows, img_cols)
img2, _, corners = generate_data(img2,img_rows, img_cols)
            #print (img1.shape)
            #print (im2d.shape)
im2d[:,:,0] = img1
im2d[:,:,1] = img2

corners = np.reshape(corners,(4,2))


X = []
X.append(im2d)
X = np.array(X, np.float32)
X /= 255

model = deep_homography_test()
model.load_weights(weights_path)
res = model.predict(X)
# res *= 137.712#Y_std
# res += 149.448#Y_mean
res *=30

ncorners = np.reshape(res,(4,2))
print(ncorners)

ncorners += corners
src = corners
dst = ncorners
#dst = np.array([[40,   30 ], [371,   30], [40,  220], [371, 220]], np.float32)
# temp = dst
# dst = src
# src = temp

print('src:',src.dtype)
print('dst:', dst.dtype)


h = cv2.getPerspectiveTransform(src, dst)

FIELD_SHAPE = (img.shape[1] + img2.shape[1], img.shape[0])
transformed_img = cv2.warpPerspective(img2, h, FIELD_SHAPE) # change
#transformed_img[0:img.shape[0], 0:img.shape[1]] = img


fig = plt.figure()
for i in [img, img2, transformed_img]:
    fig.add_subplot(3, 1, 1)
    plt.imshow(img)
    fig.add_subplot(3,1,2)
    plt.imshow(img2)
    fig.add_subplot(3, 1, 3)
    plt.imshow(transformed_img)
plt.show()