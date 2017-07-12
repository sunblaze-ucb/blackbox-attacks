"""ResNet Evaluation module.
"""
import time
import six
import sys
import os

import cifar_distorted_input
import numpy as np
import cifar10_model
import tensorflow as tf
from tensorflow.python.platform import flags
from matplotlib import image as img

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', 'eval', 'Train or evaluate')
tf.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.flags.DEFINE_string('eval_data_path', 'cifar10', 'Filepattern for training data.')
tf.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.flags.DEFINE_integer('eval_batch_count', 100,
                            'Number of batches to eval.')
tf.flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')


def evaluate(hps, batch_size):
    """Eval loop."""
    script_dir = os.path.dirname(__file__)

    # eval_dir = FLAGS.eval_dir
    eps_num = args.eps
    eps = eps_num/255.

    #TODO: Change this based on the input! The ordering of labels is off
    # if 'cifar10_std' not in
    batch_orig_labels = np.load('cifar10' + '/batch_orig_labels.npy')
    batch_orig_images = np.load('cifar10' + '/batch_orig_images.npy')

    for i in range(10):
        img.imsave( 'images/cifar10_{}.png'.format(i),
            batch_orig_images[i].reshape(FLAGS.image_size, FLAGS.image_size, 3))

    N0, H0, W0, C0 = batch_orig_images.shape
    # N1, L1 = batch_labels.shape

    X = tf.placeholder(shape=(batch_size, H0, W0, C0), dtype=tf.float32)
    Y = tf.placeholder(shape=(batch_size), dtype=tf.float32)

    # x = tf.Variable(X, dtype=tf.float32)
    x_cropped = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 24, 24), X)
    # x_flipped = tf.map_fn(lambda img:tf.image.random_flip_left_right(img), x_cropped)
    # x_bright = tf.map_fn(lambda img:tf.image.random_brightness(img, max_delta=63), x_flipped)
    # x_contrast = tf.map_fn(lambda img:tf.image.random_contrast(img, lower=0.2, upper=1.8), x_bright)
    x_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_cropped)

    model = cifar10_model.ConvNet(hps, x_scaled, Y, FLAGS.mode, eps)
    model.build_graph()
    print('Created graph')

    loss1 = model.ben_cost
    grad = tf.gradients(loss1, X)[0]

    if args.attack == 'fgs':
        signed_grad = tf.sign(grad)
        scaled_signed_grad = eps * signed_grad
        adv_X = tf.stop_gradient(X + scaled_signed_grad)

    # Creating array to store adversarial samples
    images_adv = np.zeros((10000,32,32,3))

    # Clipping images to bounding box
    clip_min = 0
    clip_max = 255

    if (clip_min is not None) and (clip_max is not None):
        adv_X = tf.clip_by_value(adv_X, clip_min, clip_max)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    summary_writer = tf.summary.FileWriter(eval_dir)
    print('Created saver')

    sess = tf.Session()
    print('Created session')
    tf.train.start_queue_runners(sess)

    adv_exist_flag = 0
    if len(args.src_models) >= 1:
        src_models = args.src_models
        adv_exist_flag = 1
    else:
        src_models = []
        src_models.append(args.target_model)

    best_precision = 0.0
    best_precision_adv = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        for item in src_models:
            total_prediction, correct_prediction = 0, 0
            correct_adv = 0
            for i in six.moves.range(FLAGS.eval_batch_count):
                # print('{}'.format(i))
                curr_batch = batch_orig_images[i*batch_size:(i+1)*batch_size]
                curr_labels = batch_orig_labels[i*batch_size:(i+1)*batch_size]

                if adv_exist_flag == 0:
                    images_adv_curr = sess.run([adv_X], feed_dict={X: curr_batch, Y: curr_labels})
                    images_adv_curr = images_adv_curr[0]
                    images_adv[i*batch_size:(i+1)*batch_size] = images_adv_curr
                else:
                    adv_path = 'adv_samples/'+item+'_{}.npy'.format(eps_num)
                    if os.path.exists(os.path.join(script_dir,adv_path)):
                        images_adv = np.load(adv_path)
                    else:
                        raise ValueError('No adversarial samples exist for given source')
                    images_adv_curr = images_adv[i*batch_size:(i+1)*batch_size]

                if i==0:
                    for j in range(10):
                        # print(images_adv_curr[j].shape())
                        img.imsave( 'images/cifar10_adv_{}_{}_{}.png'.format(item , j, eps_num),
                            images_adv_curr[j].reshape(FLAGS.image_size, FLAGS.image_size, 3))

                (loss_adv, predictions_adv) = sess.run(
                  [model.cost, model.predictions], feed_dict={X: images_adv_curr, Y: curr_labels})

                (summaries, loss, predictions, truth, train_step) = sess.run(
                  [model.summaries, model.ben_cost, model.predictions,
                   model.labels, model.global_step], feed_dict={X: curr_batch, Y: curr_labels})

                # truth = np.argmax(truth, axis=1)
                predictions = np.argmax(predictions, axis=1)
                predictions_adv = np.argmax(predictions_adv, axis=1)
                correct_prediction += np.sum(truth == predictions)
                correct_adv += np.sum(truth == predictions_adv)
                total_prediction += predictions.shape[0]

            precision = 1.0 * correct_prediction / total_prediction
            adv_precision = 1.0 * correct_adv / total_prediction
            best_precision = max(precision, best_precision)
            best_precision_adv = max(adv_precision, best_precision_adv)

            precision_summ = tf.Summary()
            precision_summ.value.add(
                tag='Precision', simple_value=precision)
            summary_writer.add_summary(precision_summ, train_step)
            best_precision_summ = tf.Summary()
            best_precision_summ.value.add(
                tag='Best Precision', simple_value=best_precision)
            summary_writer.add_summary(best_precision_summ, train_step)
            summary_writer.add_summary(summaries, train_step)
            print('{}->{}'.format(item, args.target_model))
            tf.logging.info(tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f, adv: %.3f, best adv: %.3f, ' %
                                    (loss, precision, best_precision,1.- adv_precision, 1. - best_precision_adv)))
            summary_writer.flush()

        # Storing adversarial samples in external array
        if adv_exist_flag == 0:
            np.save('adv_samples/'+args.target_model+'_{}.npy'.format(eps_num),images_adv)

        if FLAGS.eval_once:
          break


def main():
    batch_size = 100

    # Assuming dataset is CIFAR-10
    num_classes = 10

    hps = cifar10_model.HParams(batch_size=batch_size, adv_only=False,
                             num_classes=num_classes, optimizer='mom')

    evaluate(hps, batch_size)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["fgs"])
    parser.add_argument('target_model', type=str,
                        help='name of target model')
    parser.add_argument("src_models", nargs='*',
                        help="source model for attack")
    parser.add_argument("--eps", type=int, default=8,
                        help="FGS attack scale")

    args = parser.parse_args()
    log_root = 'logs_'+args.target_model
    eval_dir = log_root+'/eval'

    main()
