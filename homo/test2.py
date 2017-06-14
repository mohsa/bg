import cv2
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

img_rows, img_cols = 180, 320

perturb_x = 10 #percent
perturb_y = 10 #percent

x_start = int((img_cols*perturb_x)/100)
y_start = int((img_rows*perturb_y)/100)

img_cols_ds = img_cols + 4*x_start
im_orig  = cv2.imread("frames_003311.png")
r = img_cols_ds / im_orig.shape[1]
dim = (img_cols_ds, (int(im_orig.shape[0] * r)))
img = cv2.cvtColor(cv2.resize(im_orig, (dim), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)



y_end = y_start + 180
x_end = x_start + 320


y_1 = y_start
x_1 = x_start
y_2 = y_end
x_2 = x_start
y_3 = y_end
x_3 = x_end
y_4 = y_start
x_4 = x_end
print(y_start)
print(y_end)
print(x_start)
print(x_end)
			# print('\n')

img_patch = img[y_start:y_end, x_start:x_end]
			# cv2.imshow('patch', img_patch)
			# cv2.waitKey(0)
print(img_patch.shape)

			# print('<===== perburbed image patch =====>')
y_1_offset = np.random.randint(-32,32)
x_1_offset = np.random.randint(-32,32)
y_2_offset = np.random.randint(-32,32)
x_2_offset = np.random.randint(-32,32)

y_3_offset = np.random.randint(-32,32)
x_3_offset = np.random.randint(-32,32)
y_4_offset = np.random.randint(-32,32)
x_4_offset = np.random.randint(-32,32)

y_1_p = y_1 + y_1_offset
x_1_p = x_1 + x_1_offset
y_2_p = y_2 + y_2_offset
x_2_p = x_2 + x_2_offset
y_3_p = y_3 + y_3_offset
x_3_p = x_3 + x_3_offset
y_4_p = y_4 + y_4_offset
x_4_p = x_4 + x_4_offset

			# print(y_p_start)
			# print(y_p_end)
			# print(x_p_start)
			# print(x_p_end)

			#img_patch_perturb = img[y_p_start:y_p_end, x_p_start:x_p_end]
			# cv2.imshow('p_patch', img_patch_perturb)
			# cv2.waitKey(0)

pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

			# print(h)
			# print(status)

img_perburb = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))

			# print(img_perburb2.shape)
			# cv2.imshow('perburb', img_perburb)
			# cv2.waitKey(0)

img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]


fig = plt.figure()


for i in [img, img_patch, img_perburb_patch]:

    fig.add_subplot(3, 1, 1)
    plt.imshow(img, cmap='gray', interpolation='bicubic')

    fig.add_subplot(3,1,2)
    plt.imshow(img_patch, cmap='gray', interpolation='bicubic')

    fig.add_subplot(3, 1, 3)
    plt.imshow(img_perburb_patch, cmap='gray', interpolation='bicubic')


plt.show()