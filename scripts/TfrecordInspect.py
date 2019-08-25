#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 22:28:52 2019

@author: user1
"""

import tensorflow as tf

tf_record = '../data/processed/random-42/train/001.tfrecord' 

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

dataset = tf.data.TFRecordDataset(tf_record,compression_type="GZIP")
iterator = dataset.make_initializable_iterator()
next_dataset = iterator.get_next()
single_example = tf.parse_single_example(next_dataset, features=keys_to_features)

CELL_TYPES = {'HEPG2':0,'HUVEC':1,'RPE':2,'U2OS':3}
CELL_keys = list(CELL_TYPES.keys())
CELL_values = [CELL_TYPES[k] for k in CELL_keys]
Cell_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(CELL_keys, CELL_values, key_dtype=tf.string, value_dtype=tf.int32), -1
        )
    
with tf.Session() as sess:
    sess.run([iterator.initializer,tf.tables_initializer()])
    record = sess.run(single_example)
    plate = record['plate']
    exp = record['experiment']
    one_hot_plate = tf.one_hot(plate-1, 4)
    onp = sess.run(one_hot_plate)
    cell = sess.run(Cell_table.lookup(single_example['cell_type']))
    

type('HEPG2')

if record["cell_type"].decode('UTF-8')=='HUVEC':
    print('yes')

