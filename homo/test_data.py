import numpy as np
import json
import os
import cv2
import datetime
from random import shuffle
#from deep_homography_estimation_dg import get_data_batches

max_dataset_size = 700
validation_start = 25
batch_size = 32
nb_epoch = 10

image_folder = '../Data/Images/frames_360p'
samples = []
img_rows, img_cols = 180, 320
img_cols_ds = 448


def data_batches(img, h, w, offsets):
  img_rows, img_cols = (h, w)
  img = img

  perturb_x = 10  # percent
  perturb_y = 10  # percent

  x_start = int((img_cols * perturb_x) / 100)
  y_start = int((img_rows * perturb_y) / 100)


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


  img_patch = img[y_start:y_end, x_start:x_end]


  # print('<===== perburbed image patch =====>')
  y_1_offset = offsets[0]
  x_1_offset = offsets[1]
  y_2_offset = offsets[2]
  x_2_offset = offsets[3]

  y_3_offset = offsets[4]
  x_3_offset = offsets[5]
  y_4_offset = offsets[6]
  x_4_offset = offsets[7]

  y_1_p = y_1 + y_1_offset
  x_1_p = x_1 + x_1_offset
  y_2_p = y_2 + y_2_offset
  x_2_p = x_2 + x_2_offset
  y_3_p = y_3 + y_3_offset
  x_3_p = x_3 + x_3_offset
  y_4_p = y_4 + y_4_offset
  x_4_p = x_4 + x_4_offset


  pts_img_patch = np.array([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]]).astype(np.float32)
  pts_img_patch_perturb = np.array([[y_1_p, x_1_p], [y_2_p, x_2_p], [y_3_p, x_3_p], [y_4_p, x_4_p]]).astype(np.float32)
  h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)


  img_perburb = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))

  img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]

  h_4pt = np.array([y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]).astype(np.int)

  #h_4pt = np.array([y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4]).astype(np.float32)
  #h_4pt = np.array([y_1_p, x_1_p, y_2_p, x_2_p, y_3_p, x_3_p, y_4_p, x_4_p]).astype(np.int)

  return img_patch, img_perburb_patch, h_4pt

def get_data_batches(bt, image_folder='../Data/Images/frames_360p'):

    X = []
    Y = []
    im2d = np.zeros((img_rows, img_cols, 2), np.uint8)

    for i in range(len(bt)):
        filename1 = bt[i][0]
        offsets = bt[i][1]

        img = cv2.imread(os.path.join(image_folder, filename1))
        r = img_cols_ds / img.shape[1]
        dim = (img_cols_ds, int(img.shape[0] * r))
        img = cv2.cvtColor(cv2.resize(img, (dim), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        count = 0
        while count < 1:
            img1, img2, corners = data_batches(img, img_rows, img_cols, offsets)
            im2d[:, :, 0] = img1
            im2d[:, :, 1] = img2

            X.append(im2d)
            Y.append(corners)
            count += 1

    X = X[:len(bt)]
    Y = Y[:len(bt)]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    X /= 255
    Y /= 10
    Y = np.round(Y,2)
    X_train = X[:len(bt)]
    Y_train = Y[:len(bt)]
    X_val = X[:]
    Y_val = Y[:]

    return X_train, Y_train, X_val, Y_val

for oset in range(max_dataset_size):
        y_1_offset = (np.random.randint(-1,2))*10#(-24, 24)
        x_1_offset = (np.random.randint(-1,2))*10
        y_2_offset = (np.random.randint(-1,2))*10
        x_2_offset = (np.random.randint(-1,2))*10

        y_3_offset = (np.random.randint(-1,2))*10
        x_3_offset = (np.random.randint(-1,2))*10
        y_4_offset = (np.random.randint(-1,2))*10
        x_4_offset = (np.random.randint(-1,2))*10
        oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]

        print(oset)
        for filename in os.listdir(image_folder):
            samples.append((filename, oset))
            #print(samples)
shuffle(samples)
print(len(samples))

nb_batch = int((len(samples)) / batch_size)
#
for ep in range(nb_epoch):
    print('='*40)
    print('Epoch:', ep +1)
    print('='*40)

    for x in range(nb_batch):
        bt = samples[x * batch_size: x * batch_size + batch_size]
        print(bt)
        X_train, Y_train, X_val, Y_val = get_data_batches(bt)
        #print(X_train)
        print(Y_train[31])
        print('val',Y_val)

        print('Ep:', ep + 1, '/', nb_epoch, 'Batch:', x + 1, '/', nb_batch)
# X_train, Y_train, X_val, Y_val = get_data_batches(bt)
# print(X_train)


