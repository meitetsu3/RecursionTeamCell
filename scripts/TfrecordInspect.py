#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 22:28:52 2019

@author: user1
"""

import pandas as pd
import numpy as np

trainpd_df = pd.read_csv(r"../train_prediction.csv")
trainpd_df['score'] = np.where(trainpd_df['sirna']==trainpd_df['pred'],1,0)

trainpd_df.groupby('experiment').mean()


import tensorflow as tf

import matplotlib.pyplot as plt

tf_record = '../data/processed/random-42/train/001.tfrecord' 

for example in tf.python_io.tf_record_iterator(tf_record):
    print(tf.train.Example.FromString(example))
    
    
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

CELL_TYPES = {b'HEPG2':0,b'HUVEC':1,b'RPE':2,b'U2OS':3}
CELL_TYPES[record["cell_type"]]

[dict([a, int(x)] for a, x in b.items()) for b in CELL_TYPES]


with tf.Session() as sess:
    sess.run(iterator.initializer)
    record = sess.run(single_example)
    
if record["cell_type"].decode('UTF-8')=='HUVEC':
    print("yes")
    HEPG2
    HUVEC
    RPE
    U2OS
    cell_one_hot = tf.one_hot(record["cell_type"], 4)
    