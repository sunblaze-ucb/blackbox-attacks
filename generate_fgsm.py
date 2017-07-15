import numpy as np
import tensorflow as tf
import tqdm

import cifar_input_nostd
import resnet_model_reusable

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_float('epsilon', 8.,
                          'Strength of adversarial perturbation. (0 to 255)')

batch_size = 100
num_classes = 10

hps = resnet_model_reusable.HParams(batch_size=batch_size,
                                    num_classes=num_classes,
                                    min_lrn_rate=0.0001,
                                    lrn_rate=0.1,
                                    num_residual_units=5,
                                    use_bottleneck=False,
                                    weight_decay_rate=0.0002,
                                    relu_leakiness=0.1,
                                    optimizer='mom')

images, labels = cifar_input_nostd.build_input(
    'cifar10', FLAGS.train_data_path, batch_size, 'eval')
images_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images)


adv_model = resnet_model_reusable.ResNet(hps, images_scaled, None, 'eval')
adv_model._build_model()
# Predict labels to use in no-label-leaking adversarial examples. Not in training mode.
adv_model.labels = tf.one_hot(tf.argmax(adv_model.logits, axis=1), depth=hps.num_classes)
# Generate adversarial examples. Not in training mode.
adv_model._build_cost()
grads, = tf.gradients(adv_model.cost, images, name='gradients_fgsm')
perturbation = FLAGS.epsilon * tf.sign(grads)
adv_images = tf.stop_gradient(tf.clip_by_value(images + perturbation, 0., 255.))

sess = tf.Session()
tf.train.start_queue_runners(sess)

saver = tf.train.Saver()
ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
saver.restore(sess, ckpt_state.model_checkpoint_path)

adv_imgs = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
adv_imgs_batches = adv_imgs.reshape(500, batch_size, 32, 32, 3)
for i in tqdm.trange(500):
    adv_imgs_batches[i] = sess.run(adv_images)

np.save('static_adv_thin.npy', adv_imgs)
