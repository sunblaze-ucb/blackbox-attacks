import numpy as np
import tensorflow as tf
import tqdm

import cifar_input_ensemble

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_data_path', 'cifar-10-batches-bin/data_batch*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_float('epsilon', 8.,
                          'Strength of adversarial perturbation. (0 to 255)')

batch_size = 100

images, images_adv_thin, labels = cifar_input_ensemble.build_input(
    'cifar10', FLAGS.train_data_path, batch_size, 'eval')

# Check that shapes and dtypes are the same.
print 'images', images, images_adv_thin

# Check that images are really within epsilon away.
correct = tf.reduce_all(tf.less_equal(images - images_adv_thin, FLAGS.epsilon))

sess = tf.Session()
tf.train.start_queue_runners(sess)

# Loop over the training set twice.
for i in tqdm.trange(1000):
    ok = sess.run(correct)
    if not ok:
        raise i
