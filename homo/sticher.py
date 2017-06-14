import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'black'
import json
import collections

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

import glob
import cv2
import tqdm
import numpy as np
from PIL import Image
from deep_homography_estimation import deep_homography_net
from homography_generator import HomographyGenerator
from constants import *
import datetime
now = datetime.datetime.now

import os
output_folder = 'output'


output_folder = 'output'
weights_path = os.path.join(output_folder, 'weights.h5')
history_path = os.path.join(output_folder, 'history.json')
stats_path = os.path.join(output_folder, 'stats.txt')
output_path = os.path.join(output_folder, 'output.json')


def get_panorama_data(image_folder='Data/Images/Merisa'):
    homography_generator = HomographyGenerator()
    annotations = homography_generator.get_ground_truth(output_format='corner', only_nfl=True)
    

    filenames = []
    X = []
    Y = []
    img_rows, img_cols = 180, 320
    im2d = np.zeros((img_rows,img_cols,2), np.uint8)
    

    for filename1, value in annotations.items():
        for filename_ref, corners in value.items():
            finename_ref = filename_ref 
            corners = corners  

        filenames.append(filename1)
        #filenames.append(filename_ref)        
                  
        img1 = cv2.imread(os.path.join(image_folder, filename1))
        img1 = cv2.cvtColor(cv2.resize(img1, (img_cols, img_rows)), cv2.COLOR_BGR2GRAY) 
        img2 = cv2.imread(os.path.join(image_folder, filename_ref))
        img2 = cv2.cvtColor(cv2.resize(img2, (img_cols, img_rows)), cv2.COLOR_BGR2GRAY) 
        
        im2d[:,:,0] = img1
        im2d[:,:,1] = img2

        X.append(im2d)
        
    
    X = np.array(X, np.float32)    

    X /= 255

        
    return X, filenames 


def stich_frames(image_folder='Data/Images/Merisa'):
    i = 1
    annotations_filename='output/output.json'
    with open(annotations_filename) as annotations_file:
      annotation = json.load(annotations_file)
      filenames_corners = collections.OrderedDict(sorted(annotation.items(), key=lambda t: t[0], reverse=True))
      


    filename_last , corners = list(filenames_corners.items())[0]
    
    img2 = cv2.imread(os.path.join(image_folder, filename_last))
    #img2 = cv2.resize(img2, (IMAGE_WIDTH, IMAGE_HEIGHT))

    for filename_ref, corners_next in list(filenames_corners.items())[1:2]:
    
       plt.figure(i)
       i += 1
       corners = np.array([(x, y) for x, y in zip(corners[0::2], corners[1::2])])
       corners += np.array(ORIGINAL_CORNERS)
       src = np.array(ORIGINAL_CORNERS, np.float32)
       dst = np.array(corners, np.float32)
       h = cv2.getPerspectiveTransform(src, dst)
       img1 =  cv2.imread(os.path.join(image_folder,filename_ref))

       FIELD_SHAPE = (img1.shape[1] + img2.shape[1], img1.shape[0])
       transformed_img = cv2.warpPerspective(img2, h, FIELD_SHAPE) # change 

       transformed_img[0:img1.shape[0], 0:img1.shape[1]] = img1


       img2 = transformed_img
       corners = corners_next
       plt.imshow(transformed_img, alpha=0.6)
       plt.show(block=False)
    return transformed_img


def predict_Homography():
    X, filenames = get_panorama_data()

    model = deep_homography_net()
    model.load_weights(weights_path)

    t = now()
    res = model.predict(X)
    print(now() - t)
    
    output = dict(zip(filenames, res.tolist()))
    with open(output_path, 'w') as outfile:
        json.dump(output, outfile)

    return output


def generate_panorama(input_data='../datasets/line_dataset/',
                            yard_heatmaps_dir='../datasets/res_yard/',
                            hash_heatmaps_dir='../datasets/res_hash/',
                            number_classif_weights='../models/number_classifier.h5',
                            output_file='line_demo_2.mov',
                            verbose=0):

    X = get_panorama_data()

    res = predict_Homography()
    imgg = stich_frames()
    print(imgg)

    return imgg
    

if __name__ == '__main__':
    generate_panorama()