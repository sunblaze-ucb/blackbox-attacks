import sys

import numpy as np

# in this one, 'a' is always the ground truth
# usage: python compare_preds_2.py b.npy [a_offset]

# Criteria: f(x') != y

a = np.argmax(np.load('test_labels.npy'), axis=1)
b = np.load(sys.argv[1])

if len(sys.argv) >= 3:
    offset = int(sys.argv[2])
    count = len(b)
    a = a[offset:offset+count]
else:
    offset = None

assert len(a) == len(b)
usethese = np.not_equal(a, b)
mismatch = np.count_nonzero(usethese)
print mismatch, 'mismatch', float(mismatch) / len(a)

# haaaaack
import re
attack_name = re.match(r'preds_(\w+_\d+)_(\w+).npy', sys.argv[1]).group(1)
ia = np.load('test_orig.npy')
ib = np.load('test_%s.npy' % attack_name)
if offset is not None:
    ia = ia[offset:offset+count]
assert ia.shape == ib.shape
rms = np.sqrt(np.sum(np.square(ib - ia), axis=(1, 2, 3)))
l2a = np.mean(rms)
l2s = np.mean(rms[usethese])
diff = '%+.2f%%' % ((l2s - l2a) / l2a * 100)
print 'l2 (all, successful)', l2a, l2s, diff
