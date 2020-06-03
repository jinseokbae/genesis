import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import os
import cv2
import pdb


# for train
total_count = 0
num_shards = 2160
data_prfx = '/home/jsbae/ClonedRepo/genesis/data/gqn_datasets/cophy_balls/train/'
ball_count = np.zeros(5)

# for test
# total_count = 1350000
# num_shards = 240
# data_prfx = '/home/jsbae/ClonedRepo/genesis/data/gqn_datasets/cophy_balls/test/'
# ball_count = np.ones(5)*270000


def get_data_prefix():
  root_dir = os.getcwd()
  return  os.path.join('/home/jsbae/ClonedRepo/cophy/images/cophy_balls/') # Separate train & test folders

def reduce_func(key, dataset):
    filename = tf.strings.join([data_prfx, tf.strings.as_string(key, width=len(str(num_shards)), fill='0'), '-of-',
                                tf.strings.as_string(num_shards), '.tfrecord'])
    # pdb.set_trace()
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

ball_count = np.zeros((5))

for shard in range(num_shards):
    batch_images = np.zeros((625,64,64,3))

    for i in range(625):
        # ball_num = 2 + total_count // 10000*30
        # episode = (total_count//30) % 10000
        # currentframe = total_count % 30
        ball_num = total_count % 5 + 2
        episode = ball_count[ball_num-2] // 30
        currentframe = ball_count[ball_num-2] % 30

        path = '/home/jsbae/ClonedRepo/cophy/images/cophy_balls/{ballN}/{epiN}/{ballN}_{epiN}_{curN}.jpg'.format(ballN = int(ball_num), epiN = int(episode), curN = int(currentframe))
        print(path)
        batch_images[i] = cv2.imread(path)
        batch_images[i] /= 255.
        total_count += 1
        ball_count[ball_num-2] += 1

    # pdb.set_trace()
    dataset = tf.data.Dataset.from_tensor_slices(batch_images)
    dataset = dataset.map(tf.io.serialize_tensor)
    # pdb.set_trace()
    dataset = dataset.enumerate()
    dataset = dataset.apply(tf.data.experimental.group_by_window(
        lambda x, _: 1 + shard, reduce_func, tf.int64.max
    ))


    # call each batch to save as file.
    for elem in dataset:
      elem