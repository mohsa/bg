import json
import os
import cv2
import time
import numpy as np
from datetime import datetime
from random import shuffle

import tensorflow as tf

#==================================================================
output_folder = '../output'
train_samples_path = os.path.join(output_folder, 'sampt')
val_samples_path = os.path.join(output_folder, 'sampv')
output_path = os.path.join(output_folder, 'output.json')

now = datetime.now()

logdir = "tf_logs/deepnn-e100-bt8-{0}".format(now.strftime("%Y%m%d-%H%M%S"))
model_path = os.path.join('latest_model', 'DeepHomographyModel')



image_folder = '../Data/Images/frames_360p'
max_data_size = 2000  # total samples are equal to max_dataset_size*images

im2patch = .8
image_height, image_width = 360, 640
patch_height, patch_width = int(image_height * im2patch), int(image_width * im2patch)
validation_start = int(max_data_size * .8)
image_shape =(patch_height, patch_width)

max_dis = 12
batch_size = 8
nb_epoch = 100

learning_rate = 0.00001
training_dropout = 1.0

num_samples_dis = 10
num_outliers = 5

c = 16   # number of filters. In original work
ch = 2  # represent two gray scaled images for homography calculation


##########################################################################################333

def generate_samples():
    print('1')
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


def get_data(samples):
    random_list = []
    X = []
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

        coord = [y_1, x_1,y_2, x_2, y_3, x_3, y_4, x_4]

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
        coordinates.append(coord)
        p1.append(img_patch)
        p2.append(img_patch)


    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)
    coordinates = np.array(coordinates, np.float32)

    X /= 255

    # print(Y.shape)
    # for i in range(5):
    #     rn = np.random.randint(0,80)
    #     # print('rn')
    #     plt.figure()
    #     plt.subplot(2,1,1)
    #     plt.title('input image1')
    #
    #     plt.imshow(X[rn][:,:,0])
    #     plt.subplot(2, 1, 2)
    #     plt.title('input image2_distorted')
    #     plt.imshow(X[rn][:, :, 1])
    # plt.show()

    return X, Y, coordinates


#
def sticher(original_images, corners, preds):
    alpha = 0.6
    ground_truth_images = np.zeros((len(original_images), image_shape[0], image_shape[1],1), dtype=np.float32)
    predicted_images = np.zeros(ground_truth_images.shape, dtype=np.float32)

    for i in range(len(original_images)):
        image = original_images[i][:,:,0]
        prediction = preds[i].reshape((4, 2))
        original_patch_corners = corners[i].reshape((4,2))

        prediction = prediction + original_patch_corners

        h_pred, _ = cv2.findHomography(original_patch_corners, prediction)

        ground_truth_images[i][:,:,0] = original_images[i][:,:,1]
        predicted_images[i][:,:,0] = cv2.warpPerspective(image, h_pred, (patch_width, patch_height))


    return  alpha * predicted_images + (1 - alpha) * ground_truth_images

#############################################################################################

class DeepHomographyModel():
    def __init__(self, filters=16, kernel_size=3):
        self.f = filters
        self.kernel_size = kernel_size

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, patch_height, patch_width, ch], name='InputData')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 8], name='TargetData')
        self.keep_prob = tf.placeholder(tf.float32, name='KeepProb')

        conv1 = tf.contrib.layers.conv2d(self.x, self.f, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv1')
        conv2 = tf.contrib.layers.conv2d(conv1, self.f, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv2')
        pool1 = tf.contrib.layers.max_pool2d(conv2, kernel_size=[2, 2], stride=[2, 2], scope='Pool1')

        conv3 = tf.contrib.layers.conv2d(pool1, self.f * 2, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv3')
        conv4 = tf.contrib.layers.conv2d(conv3, self.f * 2, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv4')
        pool2 = tf.contrib.layers.max_pool2d(conv4, kernel_size=[2, 2], stride=[2, 2], scope='Pool2')

        conv5 = tf.contrib.layers.conv2d(pool2, self.f * 4, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv5')
        conv6 = tf.contrib.layers.conv2d(conv5, self.f * 4, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv6')
        pool3 = tf.contrib.layers.max_pool2d(conv6, kernel_size=[2, 2], stride=[2, 2], scope='Pool3')

        conv7 = tf.contrib.layers.conv2d(pool3, self.f * 8, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv7')
        conv8 = tf.contrib.layers.conv2d(conv7, self.f * 8, self.kernel_size, activation_fn=tf.nn.relu, scope='Conv8')

        flattened = tf.contrib.layers.flatten(conv8, scope='Flatten')

        fc1 = tf.contrib.layers.fully_connected(flattened, 200, activation_fn=tf.nn.relu, scope='FC1')
        do1 = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = tf.contrib.layers.fully_connected(do1, 256, activation_fn=tf.nn.relu, scope='FC2')
        do2 = tf.nn.dropout(fc2, self.keep_prob)

        with tf.name_scope('Output'):
            y_conv = tf.contrib.layers.fully_connected(do2, 8, activation_fn=None)
            y_conv = tf.multiply(y_conv, 200, name='pred')

        self.loss = tf.losses.absolute_difference(y_conv, self.y_, scope='Loss')
        tf.summary.scalar('Mean absolute error', self.loss)

        self.mse = tf.losses.mean_squared_error(y_conv, self.y_, scope='MSE')
        tf.summary.scalar('Mean squared error', self.mse)

        self.merged_summary_op = tf.summary.merge_all()

        self.pred = y_conv
        #
        self.original_corners = tf.placeholder(tf.float32, name='corners')
        self.overlay_imgs = tf.py_func(sticher, [self.x ,self.original_corners, self.pred], tf.float32)
        self.imagev_summary_op = tf.summary.image('Stiched_val', self.overlay_imgs, max_outputs=15)
        self.imaget_summary_op = tf.summary.image('Stiched_train', self.overlay_imgs, max_outputs=15)
        #

        with tf.name_scope('Optimizer'):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


###########################################################################################################3
def train_net():
    samples_train, samples_val = generate_samples()

    val_samp_dis = []
    train_samp_dis = []

    for i in range(num_samples_dis):
        val_samp_dis.append(samples_val[np.random.randint(0, len(samples_val))])
        train_samp_dis .append(samples_train[np.random.randint(0, len(samples_train))])

    X_val, Y_val, _ = get_data(samples_val)

    model = DeepHomographyModel()
    model.build_model()
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


        losses = []
        mses = []
        for x in range(nb_batch):
            bt = samples_train[x * batch_size: x * batch_size + batch_size]
            X_train, Y_train, _ = get_data(bt)

            _, loss, mse = sess.run([model.train_step, model.loss, model.mse],
                                    feed_dict={model.x: X_train, model.y_: Y_train,
                                               model.keep_prob: training_dropout})

            losses.append(loss * batch_size)
            mses.append(mse * batch_size)


        train_loss = (np.array(losses).sum() / len(samples_train)).item()
        train_mse = (np.array(mses).sum() / len(samples_train)).item()
        print('mae:',train_loss)
        print('mse:', train_mse)

        if ep % 20 == 0:
            saver.save(sess, os.getcwd(), global_step=ep)

        train_loss_summary = tf.Summary()
        train_loss_summary.value.add(tag="Mean_absolute_error", simple_value=train_loss)
        train_writer.add_summary(train_loss_summary, ep)

        train_loss_summary = tf.Summary()
        train_loss_summary.value.add(tag="Mean_squared_error", simple_value=train_mse)
        train_writer.add_summary(train_loss_summary, ep)

        # Show results for some randomly selected samples from validation set
        Xv, _, Cv = get_data(samples_val)

        val_image_summary = sess.run(model.imagev_summary_op,
                                       feed_dict={model.x: Xv, model.original_corners: Cv, model.keep_prob: 1.0})
        val_writer.add_summary(val_image_summary, ep)

        Xt, _, Ct = get_data(samples_train)
        #
        train_image_summary = sess.run(model.imaget_summary_op,
                                       feed_dict={model.x: Xt, model.original_corners: Ct, model.keep_prob: 1.0})
        train_writer.add_summary(train_image_summary, ep)



        val_pred, val_summary, val_loss, val_mse = sess.run([model.pred, model.merged_summary_op, model.loss, model.mse],
            feed_dict={model.x: X_val, model.y_: Y_val, model.keep_prob: 1.0})
        val_writer.add_summary(val_summary, ep)

        print('val mae:', val_loss)
        print('val mse:', val_mse)


    saver.save(sess, os.getcwd(), global_step=nb_epoch)

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