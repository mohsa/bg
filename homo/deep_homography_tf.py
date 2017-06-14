import os
import json
import random
import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.image as mpimg

from .constants import REFERENCE_POINTS, FIELD_SHAPE, ANNOTATED_GAMES, TOTAL_DIST_BETWEEN_YARD_LINES
from .utility import line_intersections, filename_to_league

image_folder = 'Data/Images'
annotations_filename = 'Data/annotations.json'

now = datetime.now()
logdir = "tf_logs/{0}".format(now.strftime("%Y%m%d-%H%M%S"))

image_width = 640
image_height = 328
channels = 3

min_target_points = 10
num_outliers = 5

epochs = 100
batch_size = 32
max_num_random_crops = 20

patch_scale = 0.9
patch_width, patch_height = int(image_width * patch_scale), int(image_height * patch_scale)

# BL - BR - TR - TL
original_patch_corners = np.array([(0, 0), (patch_width, 0), (patch_width, patch_height), (0, patch_height)])

horizontal_margin = abs(image_width - patch_width) // 2
vertical_margin = abs(image_height - patch_height) // 2

crop_points = [(0, 0),
               (image_width - patch_width, 0),
               (image_width - patch_width, image_height - patch_height),
               (0, image_height - patch_height)]


def homography_from_4pt(gt, reference_points=original_patch_corners):
    points = gt.reshape((4, 2))
    h, _ = cv2.findHomography(reference_points, points)
    return h


def get_crop(patch, gt, sx, sy):
    h = homography_from_4pt(gt)

    ex = sx + patch_width
    ey = sy + patch_height

    new_patch = patch[sy:ey, sx:ex]

    new_points = np.array([[sx, sy, 1],
                           [ex, sy, 1],
                           [ex, ey, 1],
                           [sx, ey, 1]])

    transformed_points = np.zeros((4, 2))
    for i in range(4):
        tmp = np.dot(h, new_points[i])
        tmp /= tmp[2]
        tmp = tmp[:2]
        transformed_points[i] = tmp
    transformed_points = transformed_points.reshape(8)

    return new_patch, transformed_points


def get_center_crop(img, gt):
    return get_crop(img, gt, horizontal_margin, vertical_margin)


def get_random_crop(img, gt, target_points):
    sx = random.randrange(image_width - patch_width)
    sy = random.randrange(image_height - patch_height)

    ex = sx + patch_width
    ey = sy + patch_height

    visible_points = []
    for p in target_points:
        if p[0] < sx or p[0] > ex:
            continue
        if p[1] < sy or p[1] > ey:
            continue
        visible_points.append(p)

    if len(visible_points) >= min_target_points:
        return get_crop(img, gt, sx, sy)
    return None


def mse(prediction, ground_truth):
    return ((prediction - ground_truth) ** 2).mean()


hshift = np.array([TOTAL_DIST_BETWEEN_YARD_LINES, 0, TOTAL_DIST_BETWEEN_YARD_LINES, 0, TOTAL_DIST_BETWEEN_YARD_LINES, 0,
                   TOTAL_DIST_BETWEEN_YARD_LINES, 0])


def best_ground_truth(pred, center_gt, num_ground_truths=1):
    ground_truths = [center_gt]
    for i in range(num_ground_truths):
        ground_truths.append(center_gt + (i + 1) * hshift)
        ground_truths.append(center_gt - (i + 1) * hshift)
    ground_truths = np.array(ground_truths)

    l = ((pred - ground_truths) ** 2).mean(axis=1)
    idx = np.argmin(l)
    return ground_truths[idx]


def overlay_images(original_images, ground_truths, preds):
    alpha = 0.6
    ground_truth_images = np.zeros((len(original_images), FIELD_SHAPE[1], FIELD_SHAPE[0], 3), dtype=np.float32)
    predicted_images = np.zeros(ground_truth_images.shape, dtype=np.float32)

    for i in range(len(original_images)):
        image = original_images[i]
        ground_truth = ground_truths[i].reshape((4, 2))
        prediction = preds[i].reshape((4, 2))

        h_gt, _ = cv2.findHomography(original_patch_corners, ground_truth)
        h_pred, _ = cv2.findHomography(original_patch_corners, prediction)

        ground_truth_images[i] = cv2.warpPerspective(image, h_gt, FIELD_SHAPE)
        predicted_images[i] = cv2.warpPerspective(image, h_pred, FIELD_SHAPE)

    return alpha * predicted_images + (1 - alpha) * ground_truth_images


def get_data(max_training_data=float('inf')):
    with open(annotations_filename) as annotations_file:
        annotations = json.load(annotations_file)

    X = []
    Y = []
    train_target_points = []

    X_val = []
    Y_val = []

    validation_filename = '400874529'
    for filename, annotation in annotations.items():
        if len(X) >= max_training_data and validation_filename not in filename:
            continue
        if filename_to_league(filename) != 'NFL' and not any(x in filename for x in ANNOTATED_GAMES):
            continue

        img = mpimg.imread(os.path.join(image_folder, filename))
        img = img[:image_height, :, :]

        yard_lines = annotation['yard_lines']
        hash_lines = annotation['hash_lines']

        flattened = [x for z in hash_lines + yard_lines for y in z for x in y]
        if any(x is None for x in flattened):
            continue

        yard_lines = sorted(yard_lines)
        hash_lines = sorted(hash_lines, key=lambda line: line[0][1])

        if len(yard_lines) < 2:
            continue

        if len(hash_lines) < 2:
            continue

        target_points = np.array(line_intersections(yard_lines, hash_lines))

        if len(target_points) < min_target_points:
            continue

        reference_points = np.array(REFERENCE_POINTS['NFL'][2:2 + len(target_points)])

        if len(target_points) > len(reference_points):
            raise ValueError('Too many target points')

        h, _ = cv2.findHomography(target_points, reference_points)

        transformed_corners = [np.dot(h, (x, y, 1)) for x, y in original_patch_corners]
        transformed_corners = np.array([(x / w, y / w) for x, y, w in transformed_corners])
        transformed_corners = transformed_corners.reshape(8)

        if np.abs(transformed_corners).max() > 400:
            continue

        if validation_filename in filename:
            patch, gt = get_center_crop(img, transformed_corners)
            X_val.append(patch)
            Y_val.append(gt)
        else:
            X.append(img)
            train_target_points.append(target_points)
            Y.append(transformed_corners)

    return np.array(X), np.array(Y), np.array(X_val), np.array(Y_val), train_target_points


class DeepHomographyModel():
    def __init__(self, filters=16, kernel_size=3):
        self.f = filters
        self.kernel_size = kernel_size

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, patch_height, patch_width, channels], name='InputData')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 8], name='TargetData')

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

        fc1 = tf.contrib.layers.fully_connected(flattened, 128, activation_fn=tf.nn.relu, scope='FC1')
        fc2 = tf.contrib.layers.fully_connected(fc1, 128, activation_fn=tf.nn.relu, scope='FC2')
        y_conv = tf.contrib.layers.fully_connected(fc2, 8, activation_fn=None, scope='Output')
        y_conv = tf.multiply(y_conv, 200)

        self.loss = tf.contrib.losses.mean_squared_error(y_conv, self.y_, scope='Loss')
        tf.summary.scalar('Mean squared error', self.loss)

        self.samplewise_loss = tf.reduce_mean(tf.square(tf.subtract(y_conv, self.y_)), axis=1)

        self.pred = y_conv

        self.overlay_imgs = tf.py_func(overlay_images, [self.x, self.y_, self.pred], tf.float32)
        self.image_summary_op = tf.summary.image('Overlay', self.overlay_imgs, max_outputs=15)
        self.merged_summary_op = tf.summary.merge_all()

        self.best_images = tf.placeholder(tf.float32, name='Best_images_ph')
        self.worst_images = tf.placeholder(tf.float32, name='Worst_images_ph')
        self.best_image_summary_op = tf.summary.image('Best_images', self.best_images, max_outputs=num_outliers)
        self.worst_image_summary_op = tf.summary.image('Worst_images', self.worst_images, max_outputs=num_outliers)

        with tf.name_scope('Optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


def get_augmented_data(X_train, Y_train, target_points, max_num_random_crops, shuffled=True):
    data = []
    for i in range(len(X_train)):
        patch, gt = get_center_crop(X_train[i], Y_train[i])
        data.append((patch, gt))
        for _ in range(max_num_random_crops):
            pair = get_random_crop(X_train[i], Y_train[i], target_points[i])
            if pair is None:
                continue
            data.append(pair)

    if shuffled:
        random.shuffle(data)

    return data


def train_and_eval():
    sess = tf.Session()

    X_train, center_gt_train, X_val, center_gt_val, train_target_points = get_data()
    print("Trainset: {0}, Valset: {1}".format(len(X_train), len(X_val)))

    model = DeepHomographyModel()
    model.build_model()

    train_writer = tf.summary.FileWriter(logdir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(logdir + '/validation')

    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        start = time.time()

        # 1. Get augmented data
        epoch_data = get_augmented_data(X_train, center_gt_train, train_target_points, max_num_random_crops,
                                        shuffled=True)
        num_batches = len(epoch_data) // batch_size
        print("Augmented train data: {0}".format(len(epoch_data)))

        # 3. Run train step
        losses = []
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, i * batch_size + batch_size)
            batch = epoch_data[batch_slice]

            batch_xs = np.array([batch[k][0] for k in range(batch_size)])
            batch_center_ys = np.array([batch[k][1] for k in range(batch_size)])

            if e == 0:
                # Initialize to center alignment
                best_batch_ys = batch_center_ys
            else:
                # Otherwise, run a predict and select the ground truth alignment closest to the predict.
                batch_pred = sess.run(model.pred, feed_dict={model.x: batch_xs})
                best_batch_ys = np.array(
                    [best_ground_truth(batch_pred[k], batch_center_ys[k]) for k in range(batch_size)])
            _, loss = sess.run([model.train_step, model.loss], feed_dict={model.x: batch_xs, model.y_: best_batch_ys})
            losses.append(loss * len(batch))
        train_loss = (np.array(losses).sum() / len(epoch_data)).item()

        train_loss_summary = tf.Summary()
        train_loss_summary.value.add(tag="Mean_squared_error", simple_value=train_loss)
        train_writer.add_summary(train_loss_summary, e)

        # 4. Validate
        # Get image summaries for num_samples train samples
        num_samples = 20
        train_xs = np.array([epoch_data[i][0] for i in range(num_samples)])
        train_center_ys = np.array([epoch_data[i][1] for i in range(num_samples)])

        # Get best ground truth for each sample and and write image summaries to file
        train_pred = sess.run(model.pred, feed_dict={model.x: train_xs})
        best_train_ys = np.array([best_ground_truth(train_pred[k], train_center_ys[k]) for k in range(num_samples)])
        train_image_summary = sess.run(model.image_summary_op, feed_dict={model.x: train_xs, model.y_: best_train_ys})
        train_writer.add_summary(train_image_summary, e)

        # Get total summary on entire validation set.
        val_pred = sess.run(model.pred, feed_dict={model.x: X_val})
        Y_val_moving = np.array([best_ground_truth(val_pred[i], center_gt_val[i]) for i in range(len(center_gt_val))])
        val_moving_pred, val_moving_summary, val_moving_loss, samplewise_losses = sess.run(
            [model.pred, model.merged_summary_op, model.loss, model.samplewise_loss],
            feed_dict={model.x: X_val, model.y_: Y_val_moving})
        val_writer.add_summary(val_moving_summary, e)

        sorted_indices = samplewise_losses.argsort()

        X_best_from_val = np.array([X_val[idx] for idx in sorted_indices[:num_outliers]])
        Y_best_from_val = np.array([Y_val_moving[idx] for idx in sorted_indices[:num_outliers]])
        best_images = sess.run(model.overlay_imgs, feed_dict={model.x: X_best_from_val, model.y_: Y_best_from_val})
        best_image_summary = sess.run(model.best_image_summary_op, feed_dict={model.best_images: best_images})
        val_writer.add_summary(best_image_summary, e)

        X_worst_from_val = np.array([X_val[idx] for idx in sorted_indices[:-(num_outliers + 1):-1]])
        Y_worst_from_val = np.array([Y_val_moving[idx] for idx in sorted_indices[:-(num_outliers + 1):-1]])
        worst_images = sess.run(model.overlay_imgs, feed_dict={model.x: X_worst_from_val, model.y_: Y_worst_from_val})
        worst_image_summary = sess.run(model.worst_image_summary_op, feed_dict={model.worst_images: worst_images})
        val_writer.add_summary(worst_image_summary, e)

        val_samplewise_loss = ((val_moving_pred - Y_val_moving) ** 2).mean(axis=1)
        mean = val_samplewise_loss.mean()
        std = val_samplewise_loss.std()
        median = np.median(val_samplewise_loss)

        # 5. Output status
        print(
            "epoch: {0} training loss: {1:.4f} val_loss: {2:.4f} val_median: {3:.2f} val_std: {4:.2f} time: {5:.2f}".format(
                e, train_loss, val_moving_loss, median, std, time.time() - start))

    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()


if __name__ == '__main__':
    train_and_eval()