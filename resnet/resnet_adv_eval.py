"""ResNet Train module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.platform import flags
from matplotlib import image as img

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', 'eval', 'Train or evaluate')
tf.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.flags.DEFINE_string('eval_data_path', 'cifar10/test_batch.bin', 'Filepattern for training data.')
tf.flags.DEFINE_string('log_root', 'logs',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.flags.DEFINE_string('eval_dir', 'logs/eval',
                           'Directory to keep eval outputs.')
tf.flags.DEFINE_integer('eval_batch_count', 1,
                            'Number of batches to eval.')
tf.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')


def evaluate(hps):
    """Eval loop."""
    # images, labels = cifar_input.build_input(
    #   FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)

    eval_dir = FLAGS.eval_dir
    eps = 8

    batch_images = np.load(eval_dir + '/batch_images.npy')
    batch_labels = np.load(eval_dir + '/batch_labels.npy')

    for i in range(5):
        img.imsave( 'images/cifar10_{}.png'.format(i),
            batch_images[i].reshape(FLAGS.image_size, FLAGS.image_size, 3))

    N0, H0, W0, C0 = batch_images.shape
    N1, L1 = batch_labels.shape

    print('{}, {}'.format(N0, N1))

    X = tf.placeholder(shape=(N0, H0, W0, C0), dtype=tf.float32)
    Y = tf.placeholder(shape=(N1, L1), dtype=tf.float32)

    model = resnet_model.ResNet(hps, X, Y, FLAGS.mode)
    model.build_graph()
    print('Created graph')

    loss1 = model.test_cost

    grad, = tf.gradients(loss1, X)

    signed_grad = tf.sign(grad)

    scaled_signed_grad = eps * signed_grad

    adv_X = tf.stop_gradient(X + scaled_signed_grad)

    clip_min = 0
    clip_max = 1

    if (clip_min is not None) and (clip_max is not None):
        adv_X = tf.clip_by_value(adv_X, clip_min, clip_max)

    # model_adv = resnet_model.ResNet(hps, adv_X, Y, FLAGS.mode)
    # model_adv.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
    print('Created saver')

    sess = tf.Session()
    # tf.train.start_queue_runners(sess)
    print('Created session')
    # return

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        images_adv = sess.run([adv_X], feed_dict={X: batch_images, Y: batch_labels})

        total_prediction, correct_prediction = 0, 0
        for i in six.moves.range(FLAGS.eval_batch_count):
            print('{}'.format(i))
            (summaries, loss, predictions, truth, train_step) = sess.run(
              [model.summaries, model.cost, model.predictions,
               model.labels, model.global_step], feed_dict={X: images_adv[0], Y: batch_labels})

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
          break

        # time.sleep(60)


def main():
    batch_size = 100

    # Assuming dataset is CIFAR-10
    num_classes = 10

    hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

    evaluate(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
