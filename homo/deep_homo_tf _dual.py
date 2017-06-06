import json
import os
import cv2
import time
import numpy as np
from datetime import datetime
from random import shuffle
import matplotlib.pyplot as plt

import tensorflow as tf

# ==================================================================
output_folder = '../output'
train_samples_path = os.path.join(output_folder, 'sampt')
val_samples_path = os.path.join(output_folder, 'sampv')
output_path = os.path.join(output_folder, 'output.json')

now = datetime.now()

logdir = "tf_logs/deepnn-e100-bt8-{0}".format(now.strftime("%Y%m%d-%H%M%S"))
model_path = os.path.join('latest_model', 'DeepHomographyModel')

image_folder = '../Data/Images/frames_360p'
max_data_size = 500  # total samples are equal to max_dataset_size*images

im2patch = .75
image_height, image_width = 360, 640
image_height, image_width = int(image_height / 2), int(image_width / 2)
patch_height, patch_width = int(image_height * im2patch), int(image_width * im2patch)
validation_start = int(max_data_size * .9)
image_shape = (patch_height, patch_width)

max_dis_x = 6
max_dis_y = 12

batch_size = 16
nb_epoch = 50

learning_rate = .00001
training_dropout = 1.0

num_outliers = 4

c = 16  # number of filters. In original work
ch = 1  # represent two gray scaled images for homography calculation


##########################################################################################333

def generate_samples(tipe):
    # print('1')
    image_folder = 'train'
    samples_path = os.path.join(output_folder, 'sampt')
    max_data_size = 100

    if tipe == 'val':
        image_folder = 'validation'
        samples_path = os.path.join(output_folder, 'sampv')
        max_data_size = int(max_data_size * .2)

    max_dsize_loop = int((max_data_size) / len(os.listdir(image_folder))) + 1
    print('loop size:==========', max_dsize_loop)

    samples = []
    i = 0

    random_list = []
    # repeated = 0
    while i < max_dsize_loop:

        for filename in os.listdir(image_folder):
            y_start = 12  # np.random.randint(14, 16)
            y_end = y_start + patch_height
            x_start = 18  # np.random.randint(16, 17)
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

            y_1_offset = (np.random.randint(-max_dis_y, max_dis_y))
            x_1_offset = (np.random.randint(-max_dis_x, max_dis_x))
            y_2_offset = (np.random.randint(-max_dis_y, max_dis_y))
            x_2_offset = (np.random.randint(-max_dis_x, max_dis_x))

            y_3_offset = (np.random.randint(-max_dis_y, max_dis_y))
            x_3_offset = (np.random.randint(-max_dis_x, max_dis_x))
            y_4_offset = (np.random.randint(-max_dis_y, max_dis_y))
            x_4_offset = (np.random.randint(-max_dis_x, max_dis_x))
            oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]

            if not [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset,
                    x_4_offset] in random_list:
                # print(i)
                random_list.append(oset)
                samples.append((filename, oset, coord))

                # else:
                #     repeated +=1
        i += 1

    # print('repeated:',repeated)
    shuffle(samples)
    print('total samles for' + tipe + ':', len(samples))
    # samples_train = samples[:validation_start]
    # samples_val = samples[validation_start:]

    # print('Total training samples:',len(samples_train))
    # print('Total test samples:', len(samples_val))

    with open(samples_path, 'w') as sampp:
        json.dump(samples, sampp)

    # with open(val_samples_path, 'w') as samp_val:
    #     json.dump(samples_val, samp_val)

    return samples


def get_data(samples, tipe):
    # random_list = []
    image_folder = 'train'

    if tipe == 'val':
        image_folder = 'validation'

    X = []
    X2 = []
    Y = []
    p1 = []
    p2 = []
    coordinates = []

    for i in range(len(samples)):
        filename1 = samples[i][0]
        offsets = samples[i][1]
        coord = samples[i][2]

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

        coord = [y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4]

        img = cv2.imread(os.path.join(image_folder, filename1))
        # print(img.shape)
        # r = img_cols_ds / img.shape[1]
        # dim = (img_cols_ds, int(img.shape[0] * r))
        img = cv2.cvtColor(cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2GRAY)

        img_patch = img[y_1:y_2, x_1:x_3]
        pts_img_patch = np.array([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]]).astype(np.float32)
        pts_img_patch_perturb = np.array([[y_1_p, x_1_p], [y_2_p, x_2_p], [y_3_p, x_3_p], [y_4_p, x_4_p]]).astype(
            np.float32)
        h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

        img_perburb = cv2.warpPerspective(img, h, (image_width, image_height))
        img_perburb_patch = img_perburb[y_1:y_3, x_1:x_3]
        #

        h_4pt = np.array([y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset,
                          x_4_offset])  # .astype(np.int)
        im2d = np.zeros((patch_height, patch_width, 1), np.uint8)
        im2dd = np.zeros((patch_height, patch_width, 1), np.uint8)
        im2d[:, :, 0] = img_patch
        im2dd[:, :, 0] = img_perburb_patch
        X.append(im2d)
        X2.append(im2dd)

        Y.append(h_4pt)
        coordinates.append(coord)
        p1.append(img_patch)
        p2.append(img_patch)

    X = np.array(X, np.float32)
    X2 = np.array(X2, np.float32)
    Y = np.array(Y, np.float32)
    coordinates = np.array(coordinates, np.float32)

    X /= 255

    return X,X2, Y, coordinates


#############################################################################################

class DeepHomographyModel():
    def __init__(self, filters=16, kernel_size=3):
        self.f = filters
        self.kernel_size = kernel_size

        self.x1 = tf.placeholder(tf.float32, shape=[None, patch_height, patch_width, ch], name='InputData')
        self.x2 = tf.placeholder(tf.float32, shape=[None, patch_height, patch_width, ch], name='InputData')
        self.keep_prob = tf.placeholder(tf.float32, name='KeepProb')


        with tf.variable_scope("dualChannel") as scope:
            self.o1 = self.dnet(self.x1)
            scope.reuse_variables()
            self.o2 = self.dnet(self.x2)
        self.y_ = tf.placeholder(tf.float32, shape=[None, 8], name='TargetData')
        self.loss = self.los()
        tf.summary.scalar('Mean absolute error', self.loss)
        self.merged_summary_op = tf.summary.merge_all()

        with tf.name_scope('Optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def dnet(self, x):


        conv1 = tf.contrib.layers.conv2d(x, self.f, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv1')
        conv2 = tf.contrib.layers.conv2d(conv1, self.f, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv2')

        flattened = tf.contrib.layers.flatten(conv2, scope='Flatten')

        fc1 = tf.contrib.layers.fully_connected(flattened, 128, activation_fn=tf.nn.relu, scope='FC1')

        return fc1


    def los(self):
        con = tf.concat([self.o1, self.o2], 1)
        fc2 = tf.contrib.layers.fully_connected(con, 256, activation_fn=tf.nn.relu, scope='FC2')
        fc3 = tf.contrib.layers.fully_connected(fc2, 256, activation_fn=tf.nn.relu, scope='FC3')
        # do2 = tf.nn.dropout(fc2, self.keep_prob)

        # print(do2.shape)

        with tf.name_scope('Output'):
            y_conv = tf.contrib.layers.fully_connected(fc3, 8, activation_fn=None)
            # y_conv = tf.multiply(y_conv, 50, name='pred')

        self.mse = tf.losses.mean_squared_error(y_conv, self.y_, scope='MSE')
        tf.summary.scalar('Mean squared error', self.mse)

        self.pred = y_conv
        #
        loss = tf.losses.absolute_difference(y_conv, self.y_, scope='Loss')

        return loss







###########################################################################################################3
def train_net():
    samples_train = generate_samples('train')
    samples_val = generate_samples('val')
    #


    model = DeepHomographyModel()
    # model.fnet()
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(logdir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(logdir + '/validation')
    sess.run(tf.global_variables_initializer())

    nb_batch = int((len(samples_train)) / batch_size)

    for ep in range(nb_epoch):
        start = time.time()
        print('=' * 40)
        print('Epoch:', ep + 1)
        print('=' * 40)

        losses_val = []
        mses_val = []

        losses = []
        mses = []
        for x in range(nb_batch):
            bt = samples_train[x * batch_size: x * batch_size + batch_size]
            X_train, X_train2, Y_train, _ = get_data(bt, 'train')

            # print(X_train.shape)

            _, loss, mse = sess.run([model.train_step, model.loss, model.mse],
                                    feed_dict={model.x1: X_train,model.x2 : X_train2, model.y_:Y_train,
                                               model.keep_prob: training_dropout})

            losses.append(loss * batch_size)
            mses.append(mse * batch_size)

        train_loss = (np.array(losses).sum() / len(samples_train)).item()
        train_mse = (np.array(mses).sum() / len(samples_train)).item()
        print('mae:', train_loss)
        print('mse:', train_mse)

        if ep % 15 == 0:
            saver.save(sess, os.getcwd(), global_step=ep)

        train_loss_summary = tf.Summary()
        train_loss_summary.value.add(tag="Mean_absolute_error", simple_value=train_loss)
        train_writer.add_summary(train_loss_summary, ep)

        train_loss_summary = tf.Summary()
        train_loss_summary.value.add(tag="Mean_squared_error", simple_value=train_mse)
        train_writer.add_summary(train_loss_summary, ep)
        #


        val_nb_batch = int((len(samples_val)) / batch_size)

        for y in range(val_nb_batch):
            bvt = samples_val[y * batch_size: y * batch_size + batch_size]

            X_val,X_val2, Y_val, _ = get_data(bvt, 'val')

            val_pred, val_summary, val_loss, val_mse = sess.run(
                [model.pred, model.merged_summary_op, model.loss, model.mse],
                feed_dict={model.x1: X_val,model.x2 : X_val2, model.y_: Y_val, model.keep_prob: 1.0})
            # val_writer.add_summary(val_summary, ep)


            losses_val.append(val_loss * batch_size)
            mses_val.append(val_mse * batch_size)

        val_writer.add_summary(val_summary, ep)
        val_loss = (np.array(losses_val).sum() / len(samples_val)).item()
        val_mse = (np.array(mses_val).sum() / len(samples_val)).item()

        print('val mae:', val_loss)
        print('val mse:', val_mse)

    saver.save(sess, os.getcwd(), global_step=nb_epoch)
    #
    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()

    sess.close()
    tf.reset_default_graph()


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)

    train_net()
