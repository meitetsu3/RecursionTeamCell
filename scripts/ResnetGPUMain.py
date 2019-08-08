
"""
Kaggle Recursion competition - disentagling experimental noize
Modifying provided script to fit multiple GPUs.
Move to directory where rxrx is a child.
"""

import json
import os
import sys
import tensorflow as tf
from rxrx.main import main

!nvidia-smi

MODEL_DIR = r"../model/LRtest"
URL_BASE_PATH = r"../data/processed/controls/random-42/Ctrl"  #r"../data/processed/random-42" 


tf.logging.set_verbosity(tf.logging.INFO)

main(url_base_path=URL_BASE_PATH,
     model_dir=MODEL_DIR,
     train_epochs=5,
     train_batch_size=16,
     num_train_images=12194,#73030 73128? 1108*33*2, 36515*2 = 73030.
     log_step_count_epochs=100,# text log
     save_summary_steps = 1,#100
     data_format='channels_last',
     tf_precision='float32',
     n_classes=1108,
     momentum=0.8,
     weight_decay=1e-6,
     base_learning_rate=0.2)