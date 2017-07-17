import numpy as np
import tensorflow as tf
import tqdm

import cifar10_input_nostd
import cifar10_reusable

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('epsilon', 8.,
                          'Strength of adversarial perturbation. (0 to 255)')

num_classes = 10

images, labels = cifar10_reusable.inputs(eval_data=False)

logits = cifar10_reusable.inference(images)
# Predict labels to use in no-label-leaking adversarial examples.
noleak_labels = tf.stop_gradient(tf.argmax(logits, axis=1))
# Generate adversarial examples.
cost = cifar10_reusable.loss(logits, noleak_labels)
grads, = tf.gradients(cost, images, name='gradients_fgsm')
perturbation = FLAGS.epsilon * tf.sign(grads)
adv_images = tf.stop_gradient(tf.clip_by_value(images + perturbation, 0., 255.))

sess = tf.Session()
tf.train.start_queue_runners(sess)

saver = tf.train.Saver()
ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_dir)
saver.restore(sess, ckpt_state.model_checkpoint_path)

adv_imgs = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
adv_imgs_batches = adv_imgs.reshape(-1, FLAGS.batch_size, 32, 32, 3)
for i in tqdm.trange(500):
    adv_imgs_batches[i] = sess.run(adv_images)

np.save('../static_adv_tutorial.npy', adv_imgs)
with open('../static_adv_tutorial.raw', 'wb') as f:
    f.write(adv_imgs)
