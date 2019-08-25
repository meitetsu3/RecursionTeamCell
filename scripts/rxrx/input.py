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
#{"HEPG2":0,"HUVEC":1,"RPE":2,"U2OS":3}  
CELL_keys = list(CELL_TYPES.keys())
CELL_values = [CELL_TYPES[k] for k in CELL_keys]

batchval_df = pd.read_csv(r"./BatchValLookup.csv")
EXP_keys = list(batchval_df['key'])
EXP_values = [np.float32(batchval_df[batchval_df['key']==k]['batch_val'])[0] for k in EXP_keys]        

def set_shapes(batch_size, feature, labels):
    """Statically set the batch_size dimension."""
    labels.set_shape(
        labels.get_shape().merge_with(tf.TensorShape([batch_size])))
    feature["image"].set_shape(feature["image"].get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    feature["cell"].set_shape(
        feature["cell"].get_shape().merge_with(tf.TensorShape([batch_size])))
    feature["plate"].set_shape(
        feature["plate"].get_shape().merge_with(tf.TensorShape([batch_size])))
    feature["experiment"].set_shape(feature["experiment"].get_shape().merge_with(
        tf.TensorShape([batch_size, None])))
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

    if pixel_stats is not None:
        mean, std = pixel_stats
        image = (tf.cast(image, tf.float32) - mean) / std

    label = parsed["sirna"]
    Cell_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(CELL_keys, CELL_values, key_dtype=tf.string, value_dtype=tf.int32), -1
            )
    Exp_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(EXP_keys, EXP_values, key_dtype=tf.string, value_dtype=tf.float32), -1
            )    
    cell = Cell_table.lookup(parsed["cell_type"])
    plate = parsed["plate"]    
    experiment = [Exp_table.lookup(parsed["experiment"]+'1'),
                                   Exp_table.lookup(parsed["experiment"]+'2'),
                                   Exp_table.lookup(parsed["experiment"]+'3'),
                                   Exp_table.lookup(parsed["experiment"]+'4'),
                                   Exp_table.lookup(parsed["experiment"]+'5'),
                                   Exp_table.lookup(parsed["experiment"]+'6')]
                                                 
    return {"image":image,"cell":cell,"plate":plate,"experiment":experiment}, label


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
        tf.contrib.data.map_and_batch(
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
