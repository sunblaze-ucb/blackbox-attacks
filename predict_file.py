import sys

import numpy as np
import tensorflow as tf
import tqdm

import models

# usage: python predict_file.py <destination> <images_file> <ckpt_dir>

dst = sys.argv[1]
src = sys.argv[2]
ckpt_dir = sys.argv[3]

images_array = np.load(src)
batch_size = min(len(images_array), 100)
images_batches = images_array.reshape((-1, batch_size, 32, 32, 3))

images = tf.placeholder(shape=(batch_size, 32, 32, 3), dtype=tf.float32)
net = models.load_model(ckpt_dir, batch_size, images)
logits = net.get_logits()
preds = tf.argmax(logits, axis=1)
sess = tf.Session()
net.load(sess)

preds_array = np.zeros(len(images_array), dtype=np.uint8)
preds_batches = preds_array.reshape((-1, batch_size))
for i in tqdm.trange(len(preds_batches)):
    preds_batches[i] = sess.run(preds, feed_dict={
        images: images_batches[i],
    })

np.save(dst, preds_array)
