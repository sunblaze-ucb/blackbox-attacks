"""adversarial examples generation
"""
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import variables

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'eval', 'eval.')
tf.app.flags.DEFINE_string('eval_data_path', 'cifar10/test_batch.bin',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('eval_dir', 'logs/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def evaluate(hps):
  """Eval loop."""
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  tf.train.start_queue_runners(sess)

  batch_images, batch_labels = sess.run(
      [images, labels])
  np.save(FLAGS.eval_dir + '/batch_images.npy', batch_images)
  np.save(FLAGS.eval_dir + '/batch_labels.npy', batch_labels)

def main():
  # if FLAGS.num_gpus == 0:
  #   dev = '/cpu:0'
  # elif FLAGS.num_gpus == 1:
  #   dev = '/gpu:0'
  # else:
  #   raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  # with tf.device(dev):
  if FLAGS.mode == 'eval':
    print('Starting saving')
    evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run()
  main()
