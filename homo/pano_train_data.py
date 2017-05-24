import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf


X = []
Y = []

image_height, image_width = 360, 640
patch_height, patch_width = int(360 * 80 / 100), int(640 * 80 / 100)

output_folder = 'output'
samples = os.path.join(output_folder, 'sampt')

with open(samples) as samples_data:
    sdata = json.load(samples_data)
sm = np.random.randint(0,5)

filename = sdata[sm][0]
ofset = sdata[sm][1]
coord = sdata[sm][2]

# a = ["frames_003353.png", [-11, 13, 12, -2, 8, 20, -10, 9], [41, 74, 329, 74, 329, 586, 41, 586]]
# filename = a[0]
# ofset = a[1]
# coord = a[2]

image_folder = '../Data/Images/frames_360p'


img = cv2.imread(os.path.join(image_folder, filename))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

y_1 = coord[0]
x_1 = coord[1]
y_2 = coord[2]
x_2 = coord[3]
y_3 = coord[4]
x_3 = coord[5]
y_4 = coord[6]
x_4 = coord[7]

coord = [y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4]  # [lt, lb, rt, rb]

y_1_offset = ofset[0]
x_1_offset = ofset[1]
y_2_offset = ofset[2]
x_2_offset = ofset[3]
y_3_offset = ofset[4]
x_3_offset = ofset[5]
y_4_offset = ofset[6]
x_4_offset = ofset[7]
oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]

y_1_p = y_1 + y_1_offset
x_1_p = x_1 + x_1_offset
y_2_p = y_2 + y_2_offset
x_2_p = x_2 + x_2_offset
y_3_p = y_3 + y_3_offset
x_3_p = x_3 + x_3_offset
y_4_p = y_4 + y_4_offset
x_4_p = x_4 + x_4_offset

img_patch = img[y_1:y_2, x_1:x_3]
pts_img_patch = np.array([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]]).astype(np.float32)
pts_img_patch_perturb = np.array([[y_1_p, x_1_p], [y_2_p, x_2_p], [y_3_p, x_3_p], [y_4_p, x_4_p]]).astype(
    np.float32)

h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)
img_perburb = cv2.warpPerspective(img, h, (image_width, image_height))
img_perburb_patch = img_perburb[y_1:y_3, x_1:x_3]

# h_4pt = np.array([y_1, x_1, y_2, x_2, y_3, x_3, y_4, x_4]).astype(np.float32)
#h_4pt = np.array([y_1_p, x_1_p, y_2_p, x_2_p, y_3_p, x_3_p, y_4_p, x_4_p]).astype(np.int)

im2d = np.zeros((patch_height, patch_width, 2), np.uint8)

im2d[:, :, 0] = img_patch
im2d[:, :, 1] = img_perburb_patch
X.append(im2d)
Y.append(oset)
X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
X /= 255

image1 = X[0][:, :, 0]
imagep = X[0][:, :, 1]

corners_orig = np.array(coord).astype(np.float32)
corners_orig = np.reshape(corners_orig, (4,2))


sess = tf.Session()
load_path = '../'
loader = tf.train.import_meta_graph(os.path.join(load_path, 'homography-100.meta'), clear_devices=True)


loader.restore(sess, tf.train.latest_checkpoint(load_path))

# sess.run(tf.global_variables_initializer())

pred_op = tf.get_default_graph().get_tensor_by_name("Output/pred:0")

res = sess.run(pred_op, feed_dict={'InputData:0': X, 'KeepProb:0': 1.0})

print(res)
print(oset)

offsets_predicted = np.reshape(res,(4,2))
# print(offsets_predicted)

corners_new = corners_orig + offsets_predicted
src = corners_orig
dst = corners_new

# print('src:',src)
# print('dst:', dst)

h = cv2.getPerspectiveTransform(src, dst)
img_shape = (image1.shape[1] , image1.shape[0] )
transformed_img = cv2.warpPerspective(image1, h, img_shape) # change


# h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)
# imagep = cv2.warpPerspective(image1, h, img_shape)
# transformed_img[0:img.shape[0], 0:img.shape[1]] = imagep2


plt.figure()
plt.subplot(2,1,1)
plt.imshow(image1)
plt.title('given_image')#
plt.subplot(2,1,2)
plt.imshow(imagep)
plt.title('perturbed')

# plt.figure()    #
# plt.imshow(image1, cmap='gray')
# plt.imshow(transformed_img, alpha=0.6)
# plt.title(' Given vs Predicted')
alpha=0.6
# plt.figure()
# plt.imshow(imagep, cmap = 'gray')
# plt.imshow(transformed_img, alpha = 0.6)
# plt.title('GT vs predicted')
# plt.show()
plt.figure()    #

plt.imshow(alpha * imagep + (1 - alpha) * transformed_img)
plt.title(' GT vs Predicted')
plt.show()