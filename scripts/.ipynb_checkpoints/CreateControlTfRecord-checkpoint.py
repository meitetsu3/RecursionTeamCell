#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:07:04 2019

@author: user1
"""

import rxrx.io as rio
import rxrx.preprocess.images2tfrecords as i2tf 

images_path = r"../data/raw"
dest_path = r"../data/processed/controls"
meta_path = r"../data/metadata"

md = rio.combine_metadata(base_path = meta_path)
#md[(md.dataset == "train") & (md.well_type == "treatment")].drop_duplicates().sort_values(by='sirna')

#125510. 51*4*308*2=125664. 
#mdtreat = md[md.well_type == "treatment"]
#mdtest = md[md.dataset == "test"]

md.head()
md = md[md.well_type == "positive_control"]
# 12194 rows. 51*4*30*2=12240. 44 voided.
#md["dataset"]="test_pos_ctrl"

i2tf.pack_tfrecords_ctrl(images_path = images_path,
                   metadata_df = md,
                   num_workers= 12,
                   dest_path = dest_path,
                   sites_per_tfrecord=300,
                   random_seeds=[42])
