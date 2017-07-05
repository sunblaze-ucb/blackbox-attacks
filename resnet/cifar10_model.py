# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999   # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0    # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1     # Initial learning rate.

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

HParams = namedtuple('HParams',
           'batch_size, adv_only, num_classes, optimizer')


class ConvNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode, eps = None):
    """ResNet constructor.

    Args:
    hps: Hyperparameters.
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
    mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = tf.cast(labels, tf.int64)
    self.mode = mode
    self.eps = eps

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.hps.batch_size
    self.decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    self._build_model(self._images)
    self._define_ben_cost()
    if self.eps is not None:
      grad, = tf.gradients(self.ben_cost, self._images)
      signed_grad = tf.sign(grad)
      scaled_signed_grad = self.eps * signed_grad
      adv_images = tf.stop_gradient(self._images + scaled_signed_grad)
      self._build_model(adv_images, 'adv', True)
      self._define_adv_cost()
      if self.hps.adv_only == False:
        self.cost = 0.5*(self.ben_cost + self.adv_cost)
      elif self.hps.adv_only == True:
        self.cost = self.adv_cost
    else:
      self.adv_cost = tf.constant(0)
      self.cost = self.ben_cost

    tf.summary.scalar('adv_cost', self.adv_cost)
    tf.summary.scalar('cost', self.cost)

    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.summary.merge_all()


  def _build_model(self, input_var, adv=None, reuse=False):
    """Build the CIFAR-10 model.

    Args:
    images: Images returned from distorted_inputs() or inputs().

    Returns:
    Logits.
    """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
      conv = tf.nn.conv2d(input_var, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(conv1)

    # pool1
    with tf.variable_scope('pool1', reuse=reuse) as scope:
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
               padding='SAME', name='pool1')
    # norm1
    with tf.variable_scope('norm1', reuse=reuse) as scope:
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm1')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
      kernel = _variable_with_weight_decay('weights',
                       shape=[5, 5, 64, 64],
                       stddev=5e-2,
                       wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(conv2)

    # norm2
    with tf.variable_scope('norm2', reuse=reuse) as scope:
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm2')
    # pool2
    with tf.variable_scope('pool2', reuse=reuse) as scope:
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3', reuse=reuse) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2, [self.hps.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                        stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(local3)

    # local4
    with tf.variable_scope('local4', reuse=reuse) as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                        stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
      _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
      weights = _variable_with_weight_decay('weights', [192, self.hps.num_classes],
                        stddev=1/192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [self.hps.num_classes],
                  tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
      _activation_summary(softmax_linear)
      if adv is not None:
        self.adv_logits = softmax_linear
        self.adv_predictions = tf.nn.softmax(softmax_linear)
      else:
        self.logits = softmax_linear
        self.predictions = tf.nn.softmax(softmax_linear)
        self.predicted_labels = tf.reshape(tf.argmax(self.predictions, axis=1),[self.hps.batch_size])

  def _define_ben_cost(self):
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=self.logits, labels=self.labels)
    self.ben_cost = tf.reduce_mean(xent, name='xent')

  def _define_adv_cost(self):
    xent_adv = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=self.adv_logits, labels=self.predicted_labels)
    self.adv_cost = tf.reduce_mean(xent_adv, name='xent_adv')

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                  self.global_step,
                  self.decay_steps,
                  LEARNING_RATE_DECAY_FACTOR,
                  staircase=True)

    self.lrn_rate = lr
    tf.summary.scalar('learning_rate', self.lrn_rate)

    loss_averages_op = self._add_loss_summaries(self.cost)

      # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(self.lrn_rate)
      grads = opt.compute_gradients(self.cost)

      # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

      # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)

      # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

      # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
          MOVING_AVERAGE_DECAY, self.global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    # trainable_variables = tf.trainable_variables()
    # grads = tf.gradients(self.cost, trainable_variables)
    #
    # if self.hps.optimizer == 'sgd':
    #   optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    # elif self.hps.optimizer == 'mom':
    #   optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    #
    # apply_op = optimizer.apply_gradients(
    # zip(grads, trainable_variables),
    # global_step=self.global_step, name='train_step')
    #
    # train_ops = [apply_op] + self._extra_train_ops
    self.train_op = train_op


  def _add_loss_summaries(self, total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
      tf.scalar_summary(l.op.name +' (raw)', l)
      tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
  x: Tensor
  Returns:
  nothing
  """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
  name: name of the variable
  shape: list of ints
  initializer: initializer for Variable

  Returns:
  Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
  name: name of the variable
  shape: list of ints
  stddev: standard deviation of a truncated Gaussian
  wd: add L2Loss weight decay multiplied by this float. If None, weight
  decay is not added for this Variable.

  Returns:
  Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
