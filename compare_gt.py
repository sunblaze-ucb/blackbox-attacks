import sys

import numpy as np

# usage: python compare_gt.py preds_orig.npy offset preds_adv_0.npy ...

gt = np.argmax(np.load('test_labels.npy'), axis=1)
orig_preds = np.load(sys.argv[1])
offset = int(sys.argv[2])
adv_preds = np.asarray([np.load(arg) for arg in sys.argv[3:]]) # num_queries, batch

count = adv_preds.shape[1]
orig_preds = orig_preds[offset:offset+count]
gt = gt[offset:offset+count]

correct = np.equal(orig_preds, gt)
denominator = np.count_nonzero(correct)
misclassify = np.not_equal(adv_preds, gt)
cumulative_misclassify = np.cumsum(misclassify, 0)
success = np.count_nonzero(np.logical_and(correct, cumulative_misclassify), axis=1)
for i, s in enumerate(success):
    print i + 1, s, float(s) / denominator
