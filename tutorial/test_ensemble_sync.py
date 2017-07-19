import os

import numpy as np
import tensorflow as tf
import tqdm

import cifar10_input_ensemble

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '..',
                           'Path to the CIFAR-10 data directory.')
tf.app.flags.DEFINE_float('epsilon', 8.,
                          'Strength of adversarial perturbation. (0 to 255)')

batch_size = 128

images, images_adv_thin, images_adv_wide, images_adv_tutorial, labels = cifar10_input_ensemble.inputs(False, os.path.join(FLAGS.data_path, 'cifar-10-batches-bin'), batch_size)

# Check that shapes and dtypes are the same.
print 'images', images, images_adv_thin, images_adv_wide, images_adv_tutorial

# Check that images are really within epsilon away.
correct_thin = tf.reduce_all(tf.less_equal(tf.abs(images - images_adv_thin), FLAGS.epsilon))
correct_wide = tf.reduce_all(tf.less_equal(tf.abs(images - images_adv_wide), FLAGS.epsilon))
correct_tutorial = tf.reduce_all(tf.less_equal(tf.abs(images - images_adv_tutorial), FLAGS.epsilon))
correct = tf.logical_and(tf.logical_and(correct_thin, correct_wide), correct_tutorial)

sess = tf.Session()
tf.train.start_queue_runners(sess)

# Loop over the training set twice.
for i in tqdm.trange(782):
    ok = sess.run(correct)
    if not ok:
        raise i
