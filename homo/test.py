import cv2
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

img_rows, img_cols = 180, 320

###########################

perturb_x = 10 #percent
perturb_y = 10 #percent

px = int((img_cols*perturb_x)/100)
py = int((img_rows*perturb_y)/100)

print('px',px)

img_cols_ds = img_cols + 4*px


im_orig  = cv2.imread("frames_003311.png")
r = img_cols_ds / im_orig.shape[1]
dim = (img_cols_ds, (int(im_orig.shape[0] * r)))
imA = cv2.cvtColor(cv2.resize(im_orig, (dim), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

x = int((20*img_cols)/100)
#y = int()

x, y, w, h = (30, 30, 320, 180)


corners = np.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
print(imA.shape, y+h)

im_a = imA[y:y + h, x:x + w]



increments = np.random.randint(-px, px, size=(4, 2))

new_corners = corners + increments
h, _ = cv2.findHomography(new_corners, corners)
_, h_inv = cv2.invert(h)

imA_transformed = cv2.warpPerspective(imA, h_inv, (imA.shape[1], imA.shape[0]))
    # print (imA_transformed.shape, 'imA_transformed')

im_b = imA_transformed[30:210, 30:350]
    # im_b = imA_transformed[y:y+h, x:x+w]
fig = plt.figure()


for i in [imA, im_a, im_b]:

    fig.add_subplot(3, 1, 1)
    plt.imshow(imA, cmap='gray', interpolation='bicubic')

    fig.add_subplot(3,1,2)
    plt.imshow(im_a, cmap='gray', interpolation='bicubic')

    fig.add_subplot(3, 1, 3)
    plt.imshow(im_b, cmap='gray', interpolation='bicubic')


plt.show()




