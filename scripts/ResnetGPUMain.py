
"""
Kaggle Recursion competition - disentagling experimental noize
Modifying provided script to fit multiple GPUs.
Move to directory where rxrx is a child.
"""

import tensorflow as tf
from rxrx.main import main

#!nvidia-smi

MODEL_DIR = r"../model/Resnet50FAS64m015-D554WMPole-DOL1K7-bs20-ep55-CLR001-015-WD7-Cell-Welltype-ValC0318-FlipRotCrop500BiasScale01ImgChNorm-TestCtrl3"
URL_BASE_PATH = r"../data/processed/by_exp_plate_site-42"  #r"../data/processed/controls/random-42/Ctrl

tf.logging.set_verbosity(tf.logging.INFO)

main(url_base_path=URL_BASE_PATH,
     model_dir=MODEL_DIR,
     train_epochs=55,
     train_batch_size=20,
     num_train_images= 96914, #87930
     log_step_count_epochs=1,# text log
     save_summary_steps = 100,#100
     data_format='channels_last',
     n_classes=1139,
     weight_decay=1e-7,
     min_learning_rate=0.01,
     max_learning_rate=0.15)
