import sys

import numpy as np

# in this one, 'a' is always the ground truth
# usage: python compare_preds.py b.npy [a_offset]

# Criteria: f(x') != y

a = np.argmax(np.load('test_labels.npy'), axis=1)
b = np.load(sys.argv[1])

if len(sys.argv) >= 3:
    offset = int(sys.argv[2])
    count = len(b)
    a = a[offset:offset+count]

assert len(a) == len(b)
mismatch = np.count_nonzero(np.not_equal(a, b))
print mismatch, 'mismatch', float(mismatch) / len(a)
