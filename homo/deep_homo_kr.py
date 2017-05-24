import numpy as np
import json
import os
import cv2
import datetime
from random import shuffle
from matplotlib import pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras import optimizers
K.set_image_dim_ordering('tf')


#=============== Model related parameters ==================

image_folder = '../Data/Images/frames_360p'
image_height, image_width = 360, 640
max_data_size = 10000
validation_start = int((max_data_size * 80) / 100)

batch_size = 8
nb_epoch = 100

max_dis = 12 # max distortion in each corner cordinates

im2patch = .8
patch_height, patch_width = int(360*im2patch), int(640*im2patch)


c = 16   # number of filters. In original work
ch = 2  # represent two gray scaled images for homography calculation

output_folder = '../output'

train_samples_path = os.path.join(output_folder, 'sampt')
val_samples_path = os.path.join(output_folder, 'sampv')
weights_path = os.path.join(output_folder, 'weights.h5')
history_path = os.path.join(output_folder, 'history.json')
stats_path = os.path.join(output_folder, 'stats.txt')
output_path = os.path.join(output_folder, 'output.json')


# ======= Model based on paper: From https://arxiv.org/pdf/1606.03798.pdf ========

def deep_homography_test():
    model = Sequential()

    model.add(Conv2D(c, (3, 3), input_shape=(patch_height, patch_width, ch), activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Conv2D(c, (3, 3), activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Dense(256, activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Dense(8, activation='linear',kernel_initializer='glorot_uniform'))

    opt = optimizers.Adam(lr=.00001)
    model.compile(loss='mse', optimizer= opt, metrics=['mae'])

    return model


# trainiing by genereating data for whole bactch

def train():
    samples_train, samples_val = generate_samples()
    X_train, Y_train = get_data(samples_train)
    X_val, Y_val = get_data(samples_val)


    model = deep_homography_test()
    history = model.fit(X_train, Y_train, batch_size= batch_size, epochs=nb_epoch, verbose=1, validation_data=[X_val, Y_val])

    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save_weights(weights_path, overwrite=True)

    with open(history_path, 'w') as histfile:
        json.dump(history.history, histfile)


# training by generating data for each batch

def generate_samples():
    #print('1')

    max_dsize_loop = int(max_data_size / len(os.listdir(image_folder))) + 1
    print('loop size:==========', max_dsize_loop)

    samples = []
    i = 0
    max_dis = 12

    while i < max_dsize_loop:

        for filename in os.listdir(image_folder):
            y_start = np.random.randint(30, 40)
            y_end = y_start + patch_height
            x_start = np.random.randint(40, 70)
            x_end = x_start + patch_width

            y_1 = y_start
            x_1 = x_start
            y_2 = y_end
            x_2 = x_start
            y_3 = y_end
            x_3 = x_end
            y_4 = y_start
            x_4 = x_end

            coord = [y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4]

            y_1_offset = (np.random.randint(-max_dis, max_dis))  # (-24, 24)
            x_1_offset = (np.random.randint(-max_dis, max_dis))
            y_2_offset = (np.random.randint(-max_dis, max_dis))
            x_2_offset = (np.random.randint(-max_dis, max_dis))

            y_3_offset = (np.random.randint(-max_dis, max_dis))
            x_3_offset = (np.random.randint(-max_dis, max_dis))
            y_4_offset = (np.random.randint(-max_dis, max_dis))
            x_4_offset = (np.random.randint(-max_dis, max_dis))
            oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]
            samples.append((filename, oset, coord))
        i += 1

    shuffle(samples)
    print('total samles:', len(samples))
    samples_train = samples[:validation_start]
    samples_val = samples[validation_start:]

    print('Total training samples:',len(samples_train))
    print('Total test samples:', len(samples_val))

    with open(train_samples_path, 'w') as samp_train:
        json.dump(samples_train, samp_train)

    with open(val_samples_path, 'w') as samp_val:
        json.dump(samples_val, samp_val)

    return samples_train, samples_val


def train_batch():

    samples_train, samples_val = generate_samples()

    model = deep_homography_test()

    nb_batch = int((len(samples_train))/batch_size)

    for ep in range(nb_epoch):
        print('='*40)
        print('Epoch:', ep +1)
        print('='*40)

        for x in range(nb_batch):
            bt = samples_train[x * batch_size: x * batch_size + batch_size]
            print('Ep:', ep + 1, '/', nb_epoch, 'Batch:', x + 1, '/', nb_batch)
            X_train, Y_train = get_data(bt)
            loss = model.train_on_batch(X_train, Y_train)

            print('loss:', loss[0], 'mae:',loss[1])
            # print('val_acc:', lossv[2])
        X_val, Y_val = get_data(samples_val)

        loss_test = model.test_on_batch(X_val, Y_val)
        print('val_loss:', loss_test[0], 'val_mae:', loss_test[1])

    model.save_weights(weights_path, overwrite=True)

def get_data(samples):

  random_list = []
  X = []
  Y = []

  for i in range(len(samples)):
    filename1 = samples[i][0]
    offsets = samples[i][1]
    coord = samples[i][2]

    #print(filename1, offsets, coord)

    y_1 = coord[0]
    x_1 = coord[1]
    y_2 = coord[2]
    x_2 = coord[3]
    y_3 = coord[4]
    x_3 = coord[5]
    y_4 = coord[6]
    x_4 = coord[7]

    y_1_offset = offsets[0]  # (-24, 24)
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

    img = cv2.imread(os.path.join(image_folder, filename1))
    #print(img.shape)


    img = cv2.cvtColor(cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    img_patch = img[y_1:y_2, x_1:x_3]
    pts_img_patch = np.array([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]]).astype(np.float32)
    pts_img_patch_perturb = np.array([[y_1_p, x_1_p], [y_2_p, x_2_p], [y_3_p, x_3_p], [y_4_p, x_4_p]]).astype(
      np.float32)
    h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

    img_perburb = cv2.warpPerspective(img, h, (image_width, image_height))
    img_perburb_patch = img_perburb[y_1:y_3, x_1:x_3]
    #
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(img_patch)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_perburb_patch)
    # plt.show()


    # if not [y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4] in random_list:
    random_list.append([y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4])
    # h_4pt = np.array([y_1_p, x_1_p, y_2_p, x_2_p, y_3_p, x_3_p, y_4_p, x_4_p])
    h_4pt = np.array([y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset,
                      x_4_offset])  # .astype(np.int)
    im2d = np.zeros((patch_height, patch_width, 2), np.uint8)
    im2d[:, :, 0] = img_patch
    im2d[:, :, 1] = img_perburb_patch
    X.append(im2d)
    Y.append(h_4pt)


  X = np.array(X, np.float32)
  Y = np.array(Y, np.float32)

  X /= 255
  Y -= -max_dis
  Y /= (2 * max_dis)
  #print(Y)
  # print(Y)
  # for i in range(5):
  #     rn = np.random.randint(0, 8)
  #     # print('rn')
  #     plt.figure()
  #     plt.subplot(2, 1, 1)
  #     plt.title('input image1')
  #
  #     plt.imshow(X[rn][:, :, 0])
  #     plt.subplot(2, 1, 2)
  #     plt.title('input image2_distorted')
  #     plt.imshow(X[rn][:, :, 1])
  # plt.show()

  return X, Y



if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train()
    # predict()
    #train_batch()
