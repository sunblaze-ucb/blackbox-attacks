"""adversarial examples generation
"""
import time
import six
import sys

import cifar_input
import cifar_distorted_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.ops import variables

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'eval', 'eval.')
tf.app.flags.DEFINE_string('eval_data_path', 'cifar10/test_batch.bin',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('eval_data_dir', 'cifar10',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('eval_dir', 'cifar10',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')

def saver(batch_size):
  """Eval loop."""
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  if one_hot_labels == True:
    images, orig_images, labels, one_d_labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, batch_size, FLAGS.mode)
    tf.train.start_queue_runners(sess)

    batch_images, batch_orig_images, batch_labels, batch_one_d_labels = sess.run(
      [images, orig_images, labels, one_d_labels])
    np.save(FLAGS.eval_dir + '/batch_images.npy', batch_images)
    np.save(FLAGS.eval_dir + '/batch_orig_images.npy', batch_orig_images)
    np.save(FLAGS.eval_dir + '/batch_labels.npy', batch_labels)
    np.save(FLAGS.eval_dir + '/batch_orig_labels.npy', batch_one_d_labels)

  elif one_hot_labels == False:
    images, labels, orig_images = cifar_distorted_input.inputs(True, FLAGS.eval_data_dir, batch_size)
    tf.train.start_queue_runners(sess)

    batch_images, batch_labels, batch_orig_images = sess.run([images, labels, orig_images])

    np.save(FLAGS.eval_dir + '/1d_batch_labels.npy', batch_labels)
    np.save(FLAGS.eval_dir + '/orig_images.npy', batch_orig_images)

def main():
  batch_size = 10000

  num_classes = 10

  # with tf.device(dev):
  print('Starting saving')
  saver(batch_size)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run()
  one_hot_labels = True
  main()
