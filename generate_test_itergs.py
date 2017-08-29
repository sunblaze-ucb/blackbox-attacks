import sys

import numpy as np
import tensorflow as tf
import tqdm

import models

# usage: python generate_test_itergs.py <destination> <ckpt_dir> <epsilon> <step_size> <num_steps>

dest = sys.argv[1]
ckpt_dir = sys.argv[2]
epsilon = float(sys.argv[3])
step_size = float(sys.argv[4])
num_steps = int(sys.argv[5])

images_array = np.load('test_orig.npy')
labels_array = np.load('test_labels.npy')
images_batches = images_array.reshape((100, 100, 32, 32, 3))
labels_batches = labels_array.reshape((100, 100, 10))

images = tf.placeholder(shape=(100, 32, 32, 3), dtype=tf.float32)
labels = tf.placeholder(shape=(100, 10), dtype=tf.float32)
net = models.load_model(ckpt_dir, 100, images, labels)
loss = net.get_loss()
grads, = tf.gradients(loss, images, name='gradients_adv')
perturbation = step_size * tf.sign(grads)
adv_images = tf.stop_gradient(tf.clip_by_value(images + perturbation, 0., 255.))

sess = tf.Session()
net.load(sess)

adv_images_array = np.zeros((10000, 32, 32, 3), dtype=np.float32)
adv_images_batches = adv_images_array.reshape((100, 100, 32, 32, 3))
for i in tqdm.trange(100):
    clip_low = np.maximum(images_batches[i] - epsilon, 0)
    clip_high = np.minimum(images_batches[i] + epsilon, 255)
    imgs = images_batches[i]
    for step in range(num_steps):
        imgs = sess.run(adv_images, feed_dict={
            images: imgs,
            labels: labels_batches[i],
        })
    adv_images_batches[i] = np.clip(imgs, clip_low, clip_high)

np.save(dest, adv_images_array)
