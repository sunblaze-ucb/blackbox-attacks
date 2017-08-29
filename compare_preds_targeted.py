import sys

import numpy as np

# in this one, 'a' is always the target labels
# usage: python compare_preds_targeted.py b.npy [a_offset]

# Criteria: f(x') == t

a = np.argmax(np.load('test_random_targets.npy'), axis=1)
b = np.load(sys.argv[1])

if len(sys.argv) >= 3:
    offset = int(sys.argv[2])
    count = len(b)
    a = a[offset:offset+count]

assert len(a) == len(b)
match = np.count_nonzero(np.equal(a, b))
print match, 'match', float(match) / len(a)
