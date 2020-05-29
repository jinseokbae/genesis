import tensorflow as tf
import numpy as np
import os

num_shards = 240

def get_data_prefix():
  root_dir = os.getcwd()
  return  os.path.join('home/jsbae/ClonedRepo/cophy/PATH_TO_IMAGES_') # Separate train & test folders

def reduce_func(key, dataset):
    filename = tf.strings.join([data_prfx, tf.strings.as_string(key, width=len(str(num_shards)), fill='0'), '-of-',
                                tf.strings.as_string(num_shards), '.tfrecord'])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)



# call each batch to save as file.
# for elem in dataset:
#   elem
# do data manipulation

data_prfx = get_data_prefix()
dataset = tf.data.Dataset.from_tensor_slices(#tf.Tensor --> (BATCH, H, W, 3))
dataset = dataset.map(tf.io.serialize_tensor)
initial_collect_op.clear_images()

dataset = dataset.enumerate()
dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % num_shards + 1, reduce_func, tf.int64.max
))