import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import os
import cv2
import pdb
num_shards = 240
data_prfx = '/home/jsbae/ClonedRepo/genesis/data/gqn_datasets/cophy_balls/test/'
# for train
# total_count = 0

# for test
total_count = 625*9

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


for shard in range(num_shards):
    batch_images = np.zeros((625,64,64,3))

    for i in range(625):
        ball_num = 2 + total_count // 10000*30
        episode = (total_count//30) % 10000
        currentframe = total_count % 30

        path = '/home/jsbae/ClonedRepo/cophy/images/cophy_balls/{ballN}/{epiN}/{ballN}_{epiN}_{curN}.jpg'.format(ballN = ball_num, epiN = episode, curN = currentframe)
        batch_images[i] = cv2.imread(path)
        total_count += 1

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
    print('****************************')
    print('shard : ', shard, ' total count = ',total_count)
    print('****************************')
    # for train
    # if (shard % 9 == 8): total_count += 625

    # for test
    total_count += 625*9