import sys

import numpy as np
import tensorflow as tf
import tqdm

import cifar_input_nostd

# usage: python generate_test_orig.py

images, labels = cifar_input_nostd.build_input(
    'cifar10', 'cifar-10-batches-bin/test_batch.bin', 100, 'eval')

sess = tf.Session()
tf.train.start_queue_runners(sess)

images_array = np.zeros((10000, 32, 32, 3), dtype=np.float32)
labels_array = np.zeros((10000, 10), dtype=np.float32)
images_batches = images_array.reshape((100, 100, 32, 32, 3))
labels_batches = labels_array.reshape((100, 100, 10))
for i in tqdm.trange(100):
    images_batches[i], labels_batches[i] = sess.run([images, labels])

np.save('test_orig.npy', images_array)
np.save('test_labels.npy', labels_array)
