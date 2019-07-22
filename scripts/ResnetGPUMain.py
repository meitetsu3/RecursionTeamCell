
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

MODEL_DIR = r"..\model"
URL_BASE_PATH = r"..\data\processed\random-42"


tf.logging.set_verbosity(tf.logging.INFO)

main(use_tpu=False,
     tpu=None,
     gcp_project=None,
     tpu_zone=None,
     url_base_path=URL_BASE_PATH,
     use_cache=False,
     model_dir=MODEL_DIR,
     train_epochs=4,
     train_batch_size=8,
     num_train_images=73030,#73030 73128? 1108*33*2, 36515*2 = 73030.
     epochs_per_loop=1,
     log_step_count_epochs=1,
     num_cores=None,
     data_format='channels_last',
     transpose_input=False,
     tf_precision='float32',
     n_classes=1108,
     momentum=0.9,
     weight_decay=1e-4,
     base_learning_rate=0.2,
     warmup_epochs=5)