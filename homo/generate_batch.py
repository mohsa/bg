import numpy as np
import os
from random import shuffle

data_range = 1
bsize = 4
nb_batches = 4

image_folder ='../Data/Images/frames_360p'

samples = []

for oset in range(data_range):
    y_1_offset = np.random.randint(-24, 24)
    x_1_offset = np.random.randint(-24, 24)
    y_2_offset = np.random.randint(-24, 24)
    x_2_offset = np.random.randint(-24, 24)

    y_3_offset = np.random.randint(-24, 24)
    x_3_offset = np.random.randint(-24, 24)
    y_4_offset = np.random.randint(-24, 24)
    x_4_offset = np.random.randint(-24, 24)
    oset = [y_1_offset, x_1_offset, y_2_offset, x_2_offset, y_3_offset, x_3_offset, y_4_offset, x_4_offset]

    for filename in os.listdir(image_folder):

      samples.append((filename ,oset))


for x in range(nb_batches):

    bt = samples[x*bsize: x*bsize + bsize]

print(bt)
shuffle(bt)
print(bt)


print(len(samples))










