import numpy as np
import tensorflow as tf
import tqdm

import cifar_input_ensemble
import resnet_model_reusable

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')

batch_size = 100
num_classes = 10
image_size = 32

hps = resnet_model_reusable.HParams(batch_size=batch_size,
                                    num_classes=num_classes,
                                    min_lrn_rate=0.0001,
                                    lrn_rate=0.1,
                                    num_residual_units=5,
                                    use_bottleneck=False,
                                    weight_decay_rate=0.0002,
                                    relu_leakiness=0.1,
                                    optimizer='mom')

def distort_and_standardize(image):
    image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image

images, images_adv_thin, images_adv_wide, images_adv_tutorial, labels = cifar_input_ensemble.build_input('cifar10', FLAGS.train_data_path, batch_size, 'eval')

# images_scaled = tf.map_fn(distort_and_standardize, images_adv_tutorial)
images_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images_adv_tutorial)

model = resnet_model_reusable.ResNet(hps, images_scaled, None, 'eval')
model._build_model()

labels_1d = tf.argmax(labels, axis=1)
preds = tf.argmax(model.logits, axis=1)
correct = tf.count_nonzero(tf.equal(labels_1d, preds))

sess = tf.Session()
tf.train.start_queue_runners(sess)

saver = tf.train.Saver()
ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
saver.restore(sess, ckpt_state.model_checkpoint_path)

total = 0
total_correct = 0
tr = tqdm.trange(500)
for i in tr:
    total += batch_size
    total_correct += sess.run(correct)
    tr.set_postfix(total=total, total_correct=total_correct)

print 'total', total
print 'total correct', total_correct
print 'accuracy', float(total_correct) / total
