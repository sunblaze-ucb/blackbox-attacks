import sys

import numpy as np
import tqdm

import models

# usage: python generate_test_random_targets.py

labels_array = np.load('test_labels.npy')

labels_dense = np.argmax(labels_array, axis=1)
targets_dense = np.random.randint(0, 9, size=labels_dense.shape)
targets_dense += np.greater_equal(targets_dense, labels_dense)
targets_array = np.eye(10, dtype=np.float32)[targets_dense]

assert np.all(np.equal(np.sum(targets_array, axis=1), 1))
assert np.count_nonzero(labels_array * targets_array) == 0

np.save('test_random_targets.npy', targets_array)
