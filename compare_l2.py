import sys

import numpy as np

# in this one, 'a' is always the original image
# usage: python compare_l2.py b.npy [a_offset]

# This one averages over all images, regardless of success or original correctness

a = np.load('test_images.npy')
b = np.load(sys.argv[1])

if len(sys.argv) >= 3:
    offset = int(sys.argv[2])
    count = len(b)
    a = a[offset:offset+count]

assert len(a) == len(b)
rms = np.sqrt(np.sum(np.square(a - b), axis=(1, 2, 3)))

# average over all
avg = np.mean(rms)
print avg
