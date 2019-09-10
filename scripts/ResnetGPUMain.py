
"""
Kaggle Recursion competition - disentagling experimental noize
Modifying provided script to fit multiple GPUs.
Move to directory where rxrx is a child.
"""

import tensorflow as tf
from rxrx.main import main

#!nvidia-smi

MODEL_DIR = r"../model/Resnet50FAS64m00051-LL512-bs16-ep20-CLR001_02-DC5-CellPlateExp-ValCT03"
URL_BASE_PATH = r"../data/processed/by_exp_plate_site-42"  #r"../data/processed/controls/random-42/Ctrl

tf.logging.set_verbosity(tf.logging.INFO)

main(url_base_path=URL_BASE_PATH,
     model_dir=MODEL_DIR,
     train_epochs=20,
     train_batch_size=16,
     num_train_images=73030,#12194 73030 73128? 1108*33*2, 36515*2 = 73030.73030+12194=
     log_step_count_epochs=1,# text log
     save_summary_steps = 100,#100
     data_format='channels_last',
     n_classes=1108,
     weight_decay=1e-5,
     min_learning_rate=0.01,
     max_learning_rate=0.2)
