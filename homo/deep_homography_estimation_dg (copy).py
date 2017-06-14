import numpy as np
import json
import os
import cv2
import datetime
from random import shuffle

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras import optimizers
K.set_image_dim_ordering('tf')
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

from test_datagen import generate_data
from datagen_batches import data_batches

now = datetime.datetime.now


#=============== Model related parameters ==================
data_range = 100
bsize = 32
max_dataset_size = 100 # total samples are equal to max_dataset_size*images
validation_start = 600
nb_epoch = 10 #200
batch_size = 32

samples_per_epoch = 1024
img_rows, img_cols = 180, 320
img_cols_ds = 448

c = 16   # number of filters. In original work
ch = 2  # represent two gray scaled images for homography calculation

output_folder = 'output'
weights_path = os.path.join(output_folder, 'weights.h5')
history_path = os.path.join(output_folder, 'history.json')
stats_path = os.path.join(output_folder, 'stats.txt')
output_path = os.path.join(output_folder, 'output.json')


# ======= Model based on paper: From https://arxiv.org/pdf/1606.03798.pdf ========
def deep_homography_net():
    model = Sequential()
    model.add(Convolution2D(c, 3, 3, input_shape=(img_rows, img_cols, ch)))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(c, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(c*2, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(c*2, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(c * 4, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(c * 4, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(c * 8, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(c * 8, 3, 3))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    #model.add(Dropout(0.5))
    model.add(Dense(128))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128))#, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(8, activation='linear'))
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=1e-04)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    #sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    return model


#==================test model=========================
def deep_homography_test():
    model = Sequential()
    model.add(Convolution2D(c, 3, 3, input_shape=(img_rows, img_cols, ch), activation='relu'))
    model.add(Convolution2D(c, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(Convolution2D(c, 3, 3, activation='relu'))
    # model.add(Convolution2D(c, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 4, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 8, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 8, 3, 3, activation='relu'))

    model.add(Flatten())

    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(8, activation='linear'))

    adam_opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer= adam_opt, metrics=['mae', 'acc'])

    return model

#===================== Data =========================

def get_data(image_folder='../Data/Images/f2'):#frmaes_360p'):#frames_360p'):

    filenames = []
    X = []
    Y = []
    im2d = np.zeros((img_rows,img_cols,2), np.uint8)
    

    for filename1 in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename1))

        r = img_cols_ds/ img.shape[1]
        dim = (img_cols_ds, int(img.shape[0] * r))
        img = cv2.cvtColor(cv2.resize(img, (dim), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        count = 0
        while count < 1:
        
            img1, img2, corners = generate_data(img,img_rows, img_cols)        
            im2d[:,:,0] = img1
            im2d[:,:,1] = img2
            X.append(im2d)
            Y.append(corners)
            count+=1
        
    print (len(X), 'trianing samples')

    X = X[:max_dataset_size]
    Y = Y[:max_dataset_size]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    X /= 255
    Y_mean = Y.mean()
    Y_std = Y.std()
    Y /= 30

    X_train = X[:validation_start]
    Y_train = Y[:validation_start]
    X_val = X[validation_start:]
    Y_val = Y[validation_start:]
    
    return X_train, Y_train, X_val, Y_val, filenames, Y_mean, Y_std



def get_data_batches(bt, image_folder='../Data/Images/f2'):#frames_360p'):

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
    Y /= 30
    #Y = np.round(Y,2)
    X_train = X[:len(bt)]
    Y_train = Y[:len(bt)]
    X_val = X[:]
    Y_val = Y[:]

    return X_train, Y_train, X_val, Y_val


def train():
    X_train, Y_train, X_val, Y_val, _, _, _ = get_data()

    model = deep_homography_net()
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=[X_val, Y_val])

    model.save_weights(weights_path, overwrite=True)

    with open(history_path, 'w') as histfile:
        json.dump(history.history, histfile)


def train_batch():

    image_folder = '../Data/Images/f2'#frames_360p'
    samples = []

    for oset in range(max_dataset_size):
        y_1_offset = (np.random.randint(-3,4))*10#(-24, 24)
        x_1_offset = (np.random.randint(-3,4))*10
        y_2_offset = (np.random.randint(-3,4))*10
        x_2_offset = (np.random.randint(-3,4))*10

        y_3_offset = (np.random.randint(-3,4))*10
        x_3_offset = (np.random.randint(-3,4))*10
        y_4_offset = (np.random.randint(-3,4))*10
        x_4_offset = (np.random.randint(-3,4))*10
        oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]

        #print(oset)
        for filename in os.listdir(image_folder):
            samples.append((filename, oset))
            #print(samples)
    shuffle(samples)
    print(len(samples))

    model = deep_homography_test()

    nb_batch = int((len(samples))/bsize)

    for ep in range(nb_epoch):
        print('='*40)
        print('Epoch:', ep +1)
        print('='*40)

        for x in range(nb_batch):
            bt = samples[x * bsize: x * bsize + bsize]
            print('bt', bt)

            print('Ep:', ep + 1, '/', nb_epoch, 'Batch:', x + 1, '/', nb_batch)
            X_train, Y_train, X_val, Y_val = get_data_batches(bt)
            #print(X_train)
            #print(Y_train)
            # print('val:',Y_val)
            loss = model.train_on_batch(X_train, Y_train)
            #lossv = model.predict_on_batch(X_val, Y_val)

            print('loss:', loss[0], 'mae:',loss[1])
            # print('val_acc:', lossv[2])


    model.save_weights(weights_path, overwrite=True)


def predict():
    X_train, Y_train, X_val, Y_val, filenames, Y_mean, Y_std = get_data()

    X = X_train
    Y = Y_train

    model = deep_homography_net()
    model.load_weights(weights_path)

    t = now()
    res = model.predict(X)
    res *= Y_std
    res += Y_mean
    print(now() - t)

    final_mse = np.mean(np.square(Y - res), axis=None)
    with open(stats_path, 'w') as statsfile:
        print("MSE: {0}".format(final_mse), file=statsfile)
        print(str(model.to_json()), file=statsfile)

    output = dict(zip(filenames, res.tolist()))
    with open(output_path, 'w') as outfile:
        json.dump(output, outfile)


if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #train()
    # predict()
    train_batch()

