import sys

import numpy as np
import tensorflow as tf
import tqdm

import models
import opt_li

# usage: python generate_test_opt_targeted.py <destination> <ckpt_dir> <epsilon> <offset>

dest = sys.argv[1]
ckpt_dir = sys.argv[2]
epsilon = float(sys.argv[3])
offset = int(sys.argv[4])
count = 1000

images_array = np.load('test_orig.npy')[offset:offset+count]
# labels_array = np.load('test_labels.npy')[offset:offset+count]
targets_array = np.load('test_random_targets.npy')[offset:offset+count]

clip_low = np.maximum(images_array - epsilon, 0)
clip_high = np.minimum(images_array + epsilon, 255)

sess = tf.Session()
first_net = None
def m(images):
    global first_net
    net = models.load_model(ckpt_dir, 1, images, first_var=1)
    if first_net is None:
        first_net = net
    return net.get_logits()
opt = opt_li.CarliniLi(sess, m, targeted=True, learning_rate=5e-2, max_iterations=100, eps=epsilon)
first_net.load(sess)

adv_images_array = np.zeros((count, 32, 32, 3), dtype=np.float32)
for i in tqdm.trange(count):
    # .attack internally loops and calls attack_single anyway
    adv_images_array[i] = np.clip(opt.attack_single(images_array[i], targets_array[i]), clip_low[i], clip_high[i])

np.save(dest, adv_images_array)
