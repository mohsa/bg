import numpy as np
import json
import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import cv2
import collections
import datetime
now = datetime.datetime.now

#import os.path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from homography_generator import HomographyGenerator
#from homography_debugger import h_star_on_h

from keras import backend as K
K.set_image_dim_ordering('tf')

#=============== Model related parameters ==================

max_dataset_size = 45
validation_start = 40
batch_size = 32
nb_epoch = 10
samples_per_epoch = 1024
img_rows, img_cols = 180, 320

c = 8   # number of filters. In original work 
ch = 2  # represent two gray scaled images for homography calculation

output_folder = 'output'
weights_path = os.path.join(output_folder, 'weights.h5')
history_path = os.path.join(output_folder, 'history.json')
stats_path = os.path.join(output_folder, 'stats.txt')
output_path = os.path.join(output_folder, 'output.json')


# ======= Model based on paper: From https://arxiv.org/pdf/1606.03798.pdf ========
def deep_homography_net():
    model = Sequential()
    model.add(Convolution2D(c, 3, 3, input_shape=(img_rows, img_cols, ch), activation='relu'))
    model.add(Convolution2D(c, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c, 3, 3, activation='relu'))
    model.add(Convolution2D(c, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))
    model.add(Convolution2D(c * 2, 3, 3, activation='relu'))

    model.add(Flatten())

    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(8, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model

#===================== Data =========================

def get_data(image_folder='Data/Images/frames_360p'):
    homography_generator = HomographyGenerator()

    annotations = homography_generator.get_ground_truth(output_format='corner', only_nfl=True)

    filenames = []
    X = []
    Y = []
    im2d = np.zeros((img_rows,img_cols,2), np.uint8)
    

    for filename1, value in annotations.items():
        for filename2, corners in value.items():
            finename2 = filename2 
            corners = corners  

        filenames.append(filename1)
        filenames.append(filename2)
        
        #print(filename1)
        #print(filename2)
        #print('=================================')
          
        img1 = cv2.imread(os.path.join(image_folder, filename1))
        img1 = cv2.cvtColor(cv2.resize(img1, (img_cols, img_rows)), cv2.COLOR_BGR2GRAY) 
        img2 = cv2.imread(os.path.join(image_folder, filename2))
        img2 = cv2.cvtColor(cv2.resize(img2, (img_cols, img_rows)), cv2.COLOR_BGR2GRAY) 
        
        #print (img1.shape) 
        #print (im2d.shape)
        im2d[:,:,0] = img1
        im2d[:,:,1] = img2

        X.append(im2d)
        Y.append([val for point in corners for val in point])

        #img1 = img2            # updating filenames and image for second iteration
        

    X = X[:max_dataset_size]
    Y = Y[:max_dataset_size]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    X /= 255

    Y_mean = Y.mean()
    Y_std = Y.std()

    Y -= Y_mean
    Y /= Y_std
    
    X_train = X[:validation_start]
    Y_train = Y[:validation_start]
    X_val = X[validation_start:]
    Y_val = Y[validation_start:]

    
    return X_train, Y_train, X_val, Y_val, filenames, Y_mean, Y_std
    

def train():
    X_train, Y_train, X_val, Y_val, _, _, _ = get_data()

    model = deep_homography_net()
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=[X_val, Y_val])

    model.save_weights(weights_path, overwrite=True)

    with open(history_path, 'w') as histfile:
        json.dump(history.history, histfile)

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
    train()
    predict()
