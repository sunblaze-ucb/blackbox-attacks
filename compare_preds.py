import sys

import numpy as np

# usage: python compare_preds.py a.npy b.npy [a_offset]

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])

if len(sys.argv) >= 4:
    offset = int(sys.argv[3])
    count = 1000
    a = a[offset:offset+count]

assert len(a) == len(b)
changed = np.count_nonzero(np.not_equal(a, b))
print changed, 'changed', float(changed) / len(a)
