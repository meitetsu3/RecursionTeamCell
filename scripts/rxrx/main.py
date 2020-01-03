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
"""Train a ResNet-50 model on RxRx1 on GPU.

Original file:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_main.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator import estimator

from rxrx import input as rxinput
from rxrx.official_resnet import resnet_v1

DEFAULT_INPUT_FN_PARAMS = {
    'tfrecord_dataset_buffer_size': 512,
    'tfrecord_dataset_num_parallel_reads': 8,
    'parallel_interleave_cycle_length': 4,
    'parallel_interleave_block_length': 1,
    'parallel_interleave_buffer_output_elements': None,
    'parallel_interleave_prefetch_input_elements': None,
    'map_and_batch_num_parallel_calls': 8,
    'transpose_num_parallel_calls': 8,
    'prefetch_buffer_size': tf.data.experimental.AUTOTUNE,
}

# The mean and stds for each of the channels
GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,
                                10.45369445,  5.49959796, 9.81545561]),
                       np.array([7.95876312, 12.17305868, 5.86172946,
                                 7.83451711, 4.701167, 5.43130431]))

def resnet_model_fn(features,labels,mode, params, n_classes, num_train_images,train_steps,
                    data_format, train_batch_size, weight_decay, min_learning_rate, max_learning_rate,
                    model_dir,resnet_depth):
    """The model_fn for ResNet to be used with Estimator.

    Args:
    features: `Tensor` of batched images
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the Estimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.


    Returns:
        A `EstimatorSpec` for the model
    """
    if isinstance(features, dict):
        image = features['image']
        cell = features['cell']
        well_type = features['well_type']
        one_hot_cell = tf.one_hot(cell, 4)
        one_hot_well = tf.one_hot(well_type, 2)


    
    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC.
    if data_format == 'channels_first':
        image = tf.transpose(image, [0, 3, 1, 2])

    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def build_network():
        network = resnet_v1(
            resnet_depth=resnet_depth,
            num_classes=n_classes,
            data_format=data_format)
        return network(
            inputs=image, cell = one_hot_cell, well_type = one_hot_well,
            is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    one_hot_labels = tf.one_hot(labels, n_classes)
    
    global_step = tf.train.get_global_step()
    global_step_float = tf.cast(global_step,tf.float32)
    mergin = 0.15*global_step_float/train_steps
    
    original_tgt_logits = tf.reduce_sum(tf.multiply(one_hot_labels,logits),axis=1)
    merginal_tgt_logits = tf.cos(tf.acos(original_tgt_logits)+mergin)
    logitsTrain = logits-tf.multiply(one_hot_labels,logits)+tf.matmul(tf.diag(merginal_tgt_logits),one_hot_labels)
    
    logitsTrain = tf.identity(64.0*logitsTrain, 'final_dense')
    #merginal_tgt_logits = tf.math.cos(theta+0.0051)
    
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logitsTrain,
        onehot_labels=one_hot_labels)

    #d1k = tf.get_default_graph().get_tensor_by_name("dense_1/kernel:0")
    tf.logging.info("------------------------")
    #tf.logging.info(d1k)
    tf.logging.info([v.name for v in tf.trainable_variables()])

    l2loss = weight_decay*tf.add_n([
        tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name and 'deep_feature:0' not in v.name and 'Wnorm:0' not in v.name 
    ])

    tf.logging.info("l2loss: {}".format(l2loss))
    tf.logging.info("cross_entropy: {}".format(cross_entropy))
    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + l2loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        
        lrrange = max_learning_rate-min_learning_rate
        mid_step = 0.45*train_steps
        anneal_step = 0.9*train_steps

        learning_rate = tf.cond(global_step_float < mid_step,lambda:min_learning_rate+global_step_float*lrrange/mid_step
                ,lambda:tf.cond(global_step_float < anneal_step,lambda:min_learning_rate+2*lrrange-global_step_float*lrrange/mid_step
                         ,lambda:9.1*min_learning_rate-global_step_float*9*min_learning_rate/train_steps))

        #learning_rate = tf.cast(learning_rate,tf.float32)        
        #optimizer = tf.train.AdamOptimizer(learning_rate=min_learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        #gs_t = tf.reshape(global_step, [1])

        l2loss_t = tf.reshape(l2loss, [1])
        cross_entropy_t = tf.reshape(cross_entropy, [1])
        mergin_t = tf.reshape(mergin, [1])
        
        loss_t = tf.reshape(loss, [1])
        lr_t = tf.reshape(learning_rate, [1])
        
        tf.summary.scalar('l2loss', l2loss_t[0])
        tf.summary.scalar('cross_entropy', cross_entropy_t[0])
        
        #with tf.summary.create_file_writer(model_dir,max_queue=iterations_per_loop).as_default():
        tf.summary.scalar('loss', loss_t[0])
        tf.summary.scalar('learning_rate', lr_t[0])
        tf.summary.scalar('mergin', mergin_t[0])
        
    else:
        train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, axis=1)
        top_1_accuracy = tf.metrics.accuracy(labels, predictions)
        in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
        top_5_accuracy = tf.metrics.mean(in_top_5)
        tf.summary.scalar('top_1_accuracy', top_1_accuracy[1])
        tf.summary.scalar('top_5_accuracy', top_5_accuracy[1])
        eval_metrics = {'top_1_accuracy': top_1_accuracy,
                        'top_5_accuracy': top_5_accuracy}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics)
    
def main(url_base_path,
         model_dir,
         train_epochs,
         train_batch_size,
         num_train_images,
         log_step_count_epochs,
         save_summary_steps,
         data_format,
         n_classes,
         weight_decay,
         min_learning_rate,
         max_learning_rate,
         input_fn_params=DEFAULT_INPUT_FN_PARAMS,
         resnet_depth=50):

    steps_per_epoch = (num_train_images // train_batch_size)
    train_steps = steps_per_epoch * train_epochs
    current_step = estimator._load_global_step_from_checkpoint_dir(model_dir) # pylint: disable=protected-access,line-too-long
    log_step_count_steps = steps_per_epoch * log_step_count_epochs
        
    strategy = tf.distribute.MirroredStrategy()
    
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps = save_summary_steps,
        save_checkpoints_steps=steps_per_epoch,
        log_step_count_steps=log_step_count_steps,
        train_distribute = strategy)  # pylint: disable=line-too-long

    model_fn = functools.partial(
        resnet_model_fn,
        n_classes=n_classes,
        num_train_images=num_train_images,
        train_steps = train_steps,
        data_format=data_format,
        train_batch_size=train_batch_size,
        weight_decay=weight_decay,
        min_learning_rate=min_learning_rate,
        max_learning_rate=max_learning_rate,
        model_dir=model_dir,
        resnet_depth=resnet_depth)

    params = dict(batch_size=train_batch_size)
    resnet_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params)

    train_glob = os.path.join(url_base_path, 'train', '*.tfrecord')
    tf.logging.info("Train glob: {}".format(train_glob))

    eval_glob = os.path.join(url_base_path, 'val', '*.tfrecord')
    #002,012,040,test008, train016
    
    tf.logging.info("eval glob: {}".format(eval_glob))
    
    train_input_fn = functools.partial(rxinput.input_fn,
            input_fn_params=input_fn_params,
            tf_records_glob=train_glob,
            pixel_stats=GLOBAL_PIXEL_STATS)

    eval_input_fn = functools.partial(rxinput.input_fn,
            input_fn_params=input_fn_params,
            tf_records_glob=eval_glob,
            pixel_stats=GLOBAL_PIXEL_STATS)
    
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.', train_steps, train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,max_steps=train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(resnet_classifier,train_spec,eval_spec)    
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, int(time.time() - start_timestamp))


    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, elapsed_time)

    tf.logging.info('Exporting SavedModel.')

    def serving_input_receiver_fn():
        features = {
          'image': tf.placeholder(dtype=tf.float32, shape=[None, 500, 500, 6]),
          'cell': tf.placeholder(dtype=tf.int64, shape=[None]),
          'well_type': tf.placeholder(dtype=tf.int64, shape=[None]),
        }
        receiver_tensors = features
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    resnet_classifier.export_saved_model(os.path.join(model_dir, 'saved_model'), serving_input_receiver_fn)


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Train ResNet on rxrx1')
    #p.add_argument('--use-cache', type=bool, default=None)
    # Dataset Parameters
    p.add_argument(
        '--url-base-path',
        type=str,
        default='gs://rxrx1-us-central1/tfrecords/random-42',
        help=('Base path for tfrecord storage bucket url.'))
    # Training parameters
    p.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=(
            'The Google Cloud Storage bucket where the model and training summaries are'
            ' stored.'))
    p.add_argument(
        '--train-epochs',
        type=int,
        default=1,
        help=(
            'Defining an epoch as one pass through every training example, '
            'the number of total passes through all examples during training. '
            'Implicitly sets the total train steps.'))
    p.add_argument(
        '--num-train-images',
        type=int,
        default=73000
    )
    p.add_argument(
        '--train-batch-size',
        type=int,
        default=512,
        help=('Batch size to use during training.'))
    p.add_argument(
        '--n-classes',
        type=int,
        default=1108,
        help=('The number of label classes - typically will be 1108 '
              'since there are 1108 experimental siRNA classes.'))
#    p.add_argument(
#        '--epochs-per-loop',
#        type=int,
#        default=1,
#        help=('The number of steps to run on TPU before outfeeding metrics '
#              'to the CPU. Larger values will speed up training.'))
    p.add_argument(
        '--log-step-count-epochs',
        type=int,
        default=64,
        help=('The number of epochs at '
              'which global step information is logged .'))
#    p.add_argument(
#        '--num-cores',
#        type=int,
#        default=8,
#        help=('Number of TPU cores. For a single TPU device, this is 8 because '
#              'each TPU has 4 chips each with 2 cores.'))
    p.add_argument(
        '--data-format',
        type=str,
        default='channels_last',
        choices=[
            'channels_first',
            'channels_last',
        ],
        help=('A flag to override the data format used in the model. '
              'To run on CPU or TPU, channels_last should be used. '
              'For GPU, channels_first will improve performance.'))
#    p.add_argument(
#        '--transpose-input',
#        type=bool,
#        default=True,
#        help=('Use TPU double transpose optimization.'))
    p.add_argument(
        '--tf-precision',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float32'],
        help=('Tensorflow precision type used when defining the network.'))

    # Optimizer Parameters

    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument(
        '--base-learning-rate',
        type=float,
        default=0.2,
        help=('Base learning rate when train batch size is 512. '
              'Chosen to match the resnet paper.'))
    p.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
    )
    args = p.parse_args()
    args = vars(args)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Parsed args: ')
    for k, v in args.items():
        tf.logging.info('{} : {}'.format(k, v))
    main(**args)