# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient input pipeline using tf.data.Dataset.

Original file:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/imagenet_input.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial, reduce
import pandas as pd
import numpy as np
import tensorflow as tf 

CELL_TYPES = {'HEPG2':0,'HUVEC':1,'RPE':2,'U2OS':3}
CELL_keys = list(CELL_TYPES.keys())
CELL_values = [CELL_TYPES[k] for k in CELL_keys]

WELL_TYPES = {'treatment':0,'control':1}
WELL_keys = list(WELL_TYPES.keys())
WELL_values = [WELL_TYPES[k] for k in WELL_keys]

def set_shapes(batch_size, feature, labels):
    """Statically set the batch_size dimension."""
    labels.set_shape(
        labels.get_shape().merge_with(tf.TensorShape([batch_size])))
    feature["image"].set_shape(feature["image"].get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    feature["cell"].set_shape(
        feature["cell"].get_shape().merge_with(tf.TensorShape([batch_size])))
    return feature, labels


def parse_example(value,pixel_stats=None):

    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string),
        'well': tf.FixedLenFeature((), tf.string),
        'well_type': tf.FixedLenFeature((), tf.string),
        'plate': tf.FixedLenFeature((), tf.int64),
        'site': tf.FixedLenFeature((), tf.int64),
        'cell_type': tf.FixedLenFeature((), tf.string),
        'sirna': tf.FixedLenFeature((), tf.int64),
        'experiment': tf.FixedLenFeature((), tf.string)
    }

    image_shape = [512, 512, 6]
    parsed = tf.parse_single_example(value, keys_to_features)
    image_raw = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image_raw, image_shape)
    image.set_shape(image_shape)
    image = tf.image.random_crop(image,size=[500,500,6])
    b = tf.random_normal((6,), mean=0.0,stddev=0.1,seed=0)
    a = tf.random_normal((6,), mean=1.0,stddev=0.1,seed=0)   
    
    image = tf.cast(image, tf.float32)
    mean = tf.reduce_mean(image,axis=[0,1])
    std = tf.math.reduce_std(image,axis=[0,1])
    image = (image - mean) * a / std + b

#    image = tf.image.per_image_standardization(tf.cast(image, tf.float32))
    
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
   
    label = parsed["sirna"] 
    Cell_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(CELL_keys, CELL_values, key_dtype=tf.string, value_dtype=tf.int64), -1
            )
    cell = Cell_table.lookup(parsed["cell_type"])

    Well_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(WELL_keys, WELL_values, key_dtype=tf.string, value_dtype=tf.int64), -1
            )
    well = Well_table.lookup(parsed["well_type"])
    
    return {"image":image,"cell":cell, "well_type":well}, label


DEFAULT_PARAMS = dict(batch_size=512)
    
def input_fn(tf_records_glob,
             input_fn_params,
             params=None,
             pixel_stats = None,
             shuffle_buffer=64):

    batch_size = params['batch_size']
    tf.logging.info('batch_size:{}'.format(batch_size))
    
    filenames_dataset = tf.data.Dataset.list_files(tf_records_glob)

    def fetch_images(filenames):
        dataset = tf.data.TFRecordDataset(
            filenames,
            compression_type="GZIP",
            buffer_size=(1000 * 1000 *
                         input_fn_params['tfrecord_dataset_buffer_size']),
            num_parallel_reads=input_fn_params[
                'tfrecord_dataset_num_parallel_reads'])
        return dataset

    images_dataset = filenames_dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_images,
            cycle_length=input_fn_params['parallel_interleave_cycle_length'],
            block_length=input_fn_params['parallel_interleave_block_length'],
            sloppy=True,
            buffer_output_elements=input_fn_params[
                'parallel_interleave_buffer_output_elements'],
            prefetch_input_elements=input_fn_params[
                'parallel_interleave_prefetch_input_elements']))

    images_dataset = images_dataset.shuffle(2048).repeat()

    # examples dataset
    dataset = images_dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: parse_example(value,
                                        pixel_stats=pixel_stats),
            batch_size=batch_size,
            num_parallel_calls=input_fn_params['map_and_batch_num_parallel_calls'],
            drop_remainder=True))

    # Assign static batch size dimension
    dataset = dataset.map(partial(set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(
        buffer_size=input_fn_params['prefetch_buffer_size'])

    return dataset
