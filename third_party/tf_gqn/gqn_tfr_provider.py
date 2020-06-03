# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes made by ogroth, stefan.

# Adapted from https://github.com/ogroth/tf-gqn/blob/master/data_provider/gqn_tfr_provider.py
# Modified by Martin Engelcke


"""Minimal data reader for RLBench TFRecord datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import pdb

# nest = tf.contrib.framework.nest

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])

# ! The data dir is determined as follows.
# ! data/***_dataset is determined in rlbench_config.
# genesis_tf/data/rlbench_dataset/basepath
_DATASETS = dict(  # TODO: specifiy these values
    rlbench_reacher=DatasetInfo(
        basepath='panda',
        train_size=25,  # ? just # of files?
        test_size=25,  #
        frame_size=64,
        sequence_size=1),  # NOTE sequence size is trivial in our case -> overwritten by context_size.
    cophy_balls = DatasetInfo(
        basepath='cophy_balls',
        train_size = 2160,
        test_size = 240,
        frame_size = 64,
        sequence_size=1
    )
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
# --- Modified by Martin Engelcke ---
_MODES = ('train', 'test', 'devel_train', 'devel_val')


# ! --- Again modified by Cheol-Hui Min ---

# --- Modified by Martin Engelcke ---
def _get_dataset_files(dateset_info, mode, val_frac, root):
    """Generates lists of files for a given dataset version.
    @ genesis_tf/data/rlbench_datasets
    """

    basepath = dateset_info.basepath  # folder name under rlbench_dataset
    folder = 'train' if 'devel' in mode else mode
    base = os.path.join(root, basepath, folder)  # data/***_datasets/panda
    if mode == 'test':
        num_files = dateset_info.test_size
    else:
        num_files = dateset_info.train_size
    # Total # of files in train/test folder of dataset
    length = len(str(num_files))
    template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)

    if mode == 'devel_train':
        start = 0
        end = (num_files // val_frac) * (val_frac - 1)
    elif mode == 'devel_val':
        start = (num_files // val_frac) * (val_frac - 1)
        end = num_files
    else:
        start = 0
        end = num_files  # 0-of-end.tfrecord
    # data/rlbench_datasets/train/*-of-*.tfrecord

    file_list = [os.path.join(base, template.format(i + 1, num_files))
                 for i in range(start, end)]

    # print("Files for {} mode:".format(mode))
    # print(''.join(['{}\n'.format(f) for f in file_list]))

    return file_list


# ------


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class GQNTFRecordDataset(tf.data.Dataset):
    """Minimal tf.data.Dataset based TFRecord dataset.
    You can use this class to load the datasets used to train GENESIS / MONet / VAE
    in our paper;#! TILDE(Three dImensional Latent Dynamics agEnt).
    See README.md for a description of the datasets and an example of how to use
    the class.
    """

    def __init__(self, dataset, context_size, root, mode='train', val_frac=None,
                 custom_frame_size=None, num_threads=4, buffer_size=256,
                 parse_batch_size=32):
        """Instantiates a DataReader object and sets up queues for data reading.
        Args:
          dataset: string, one of ['rlbench_reacher', #TODO: many other tasks].
          #? Should I manually make the dataset to have context size?
          #!X context_size: integer, number of views to be used to assemble the context.
          root: string, path to the root folder of the data.
          mode: (optional) string, one of ['train', 'test'].
          custom_frame_size: (optional) integer, required size of the returned
              frames, defaults to None. #TODO: Check how it affects the dataset.
          num_threads: (optional) integer, number of threads used when reading and
              parsing records, defaults to 4.
          buffer_size: (optional) integer, capacity of the buffer into which
              records are read, defualts to 256.
          parse_batch_size: (optional) integer, number of records to parse at the
              same time, defaults to 32.
        Raises:
          ValueError: if the required version does not exist; if the required mode
             is not supported; if the requested context_size is bigger than the
             maximum supported for the given dataset version.
        """

        # NOTE: currently we aim to load 'rlbench_reacher'
        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, _DATASETS.keys()))

        if mode not in _MODES:  # ! one of 'train', 'test', 'devel_train', 'devel_val'
            raise ValueError('Unsupported mode {} requested. Supported modes '
                             'are {}'.format(mode, _MODES))

        self._dataset_info = _DATASETS[dataset]
        #     ? How can I prepare dataset in this format?
        #     DatasetInfo(
        #     basepath='rlbench_reacher',
        #     train_size=100000, #1e5
        #     test_size=5000, # 5e3
        #     frame_size=64,
        #     sequence_size=1

        # ? Do we use context size for training?
        # ? Maybe it relates to Latent dyanmics of SLAC.
        if context_size >= self._dataset_info.sequence_size:
            raise ValueError(
                'Maximum support context size for dataset {} is {}, but '
                'was {}.'.format(
                    dataset, self._dataset_info.sequence_size - 1, context_size))

        self._context_size = context_size  # ? Always zero?
        # Number of views in the context + target view
        self._example_size = context_size + 1  # ! 1 in our case
        self._custom_frame_size = custom_frame_size

        # ! prepare for reading a TfRecord files
        # https://www.tensorflow.org/tutorials/load_data/tfrecord
        # creates a description of the features.
        # ! we may not use this.

        # return full list of train/test files with specified path to them.
        # we need to split the dataset into train and eval
        file_names = _get_dataset_files(self._dataset_info, mode, val_frac, root)

        self._dataset = tf.data.TFRecordDataset(filenames=file_names,
                                                num_parallel_reads=num_threads)

        # ! Here's the important part.
        # prepare later elements while current element being processed.
        self._dataset = self._dataset.map(self._parse_record,  # applies mapping for each dataset
                                          num_parallel_calls=num_threads)
        self._dataset = self._dataset.prefetch(buffer_size)
        # combines consecutive elements of the dataset into batch
        self._dataset = self._dataset.batch(parse_batch_size)  # defaults to 32
        self._dataset = self._dataset.apply(tf.data.experimental.unbatch())

    def _parse_record(self, raw_data):
        """Parses the data into tensors.
           Raw data is tf.String. """

        return tf.io.parse_tensor(serialized=raw_data, out_type=tf.double)

    def _get_randomized_indices(self):
        """Generates randomized indices into a sequence of a specific length."""
        indices = tf.range(0, self._dataset_info.sequence_size)
        indices = tf.random.shuffle(indices, seed=0)
        indices = tf.slice(indices, begin=[0], size=[self._example_size])
        return indices

    def _preprocess_frames(self, example, indices):
        """Preprocesses the frames data."""
        frames = tf.concat(example, axis=0)  # concat along batch axis.
        # frames = tf.gather(frames, indices, axis=1)
        frames = tf.gather(params=frames, indices=indices, axis=1)
        dataset_image_dimensions = tuple(
            [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])  # (frame, frame, NUM_CHANNELS)
        frames = tf.reshape(
            frames, (-1, self._example_size) + dataset_image_dimensions)  # (batch, frame, frame, NUM_CHANNELS)
        if (self._custom_frame_size and
                self._custom_frame_size != self._dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(images=frames, size=new_frame_dimensions[:2],
                                     method=tf.image.ResizeMethod.BILINEAR)  # what about align_corners?
            frames = tf.reshape(
                frames, (-1, self._example_size) + new_frame_dimensions)  #
        return frames

    def _preprocess_cameras(self, example, indices):
        """Preprocesses the cameras data."""
        # ! We will not use this!

        raw_pose_params = example['cameras']
        raw_pose_params = tf.reshape(
            raw_pose_params,
            [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
        raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras

    def element_spec(self):
        return None

    # The following four methods are needed to implement a tf.data.Dataset
    # Delegate them to the dataset we create internally
    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def _inputs(self):
        return [self._dataset]

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types