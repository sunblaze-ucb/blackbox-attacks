"""ResNet Train module.
"""
import time
import six
import sys
import os

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
# tf.flags.DEFINE_string('log_root', 'logs',
#                            'Directory to keep the checkpoints. Should be a '
#                            'parent directory of FLAGS.train_dir/eval_dir.')
tf.flags.DEFINE_integer('image_size', 32, 'Image side length.')
# tf.flags.DEFINE_string('eval_dir', 'logs/eval',
#                            'Directory to keep eval outputs.')
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

    batch_images = np.load('cifar10' + '/batch_images.npy')
    batch_labels = np.load('cifar10' + '/batch_labels.npy')
    batch_orig_images = np.load('cifar10' + '/batch_orig_images.npy')

    for i in range(10):
        img.imsave( 'images/cifar10_{}.png'.format(i),
            batch_orig_images[i].reshape(FLAGS.image_size, FLAGS.image_size, 3) *255)

    N0, H0, W0, C0 = batch_images.shape
    N1, L1 = batch_labels.shape

    X = tf.placeholder(shape=(batch_size, H0, W0, C0), dtype=tf.float32)
    Y = tf.placeholder(shape=(batch_size, L1), dtype=tf.float32)

    # x = tf.Variable(X, dtype=tf.float32)
    x_scaled = tf.map_fn(lambda img: tf.image.per_image_standardization(img), X)

    model = resnet_model.ResNet(hps, x_scaled, Y, FLAGS.mode)
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
    clip_max = 1

    if (clip_min is not None) and (clip_max is not None):
        adv_X = tf.clip_by_value(adv_X, clip_min, clip_max)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(eval_dir)
    print('Created saver')

    sess = tf.Session()
    print('Created session')
    # return


    adv_exist_flag = 0
    if args.src_model is not None:
        adv_exist_flag = 1
        adv_path = 'adv_samples/'+args.src_model+'_{}.npy'.format(eps_num)
        print(adv_path)
        if os.path.exists(os.path.join(script_dir,adv_path)):
            images_adv = np.load(adv_path)
        else:
            raise ValueError('No adversarial samples exist for given source')

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

        total_prediction, correct_prediction = 0, 0
        correct_adv = 0
        for i in six.moves.range(FLAGS.eval_batch_count):

            curr_batch = batch_orig_images[i*batch_size:(i+1)*batch_size]
            curr_labels = batch_labels[i*batch_size:(i+1)*batch_size]

            if adv_exist_flag == 0:
                images_adv_curr = sess.run([adv_X], feed_dict={X: curr_batch, Y: curr_labels})
                images_adv_curr = images_adv_curr[0]
                images_adv[i*batch_size:(i+1)*batch_size] = images_adv_curr
            else:
                images_adv_curr = images_adv[i*batch_size:(i+1)*batch_size]

            if i==0:
                for j in range(10):
                    # print(images_adv_curr[j].shape())
                    img.imsave( 'images/cifar10_adv_{}_{}.png'.format(j, eps_num),
                        images_adv_curr[j].reshape(FLAGS.image_size, FLAGS.image_size, 3)*255)

            print('{}'.format(i))
            (loss_adv, predictions_adv) = sess.run(
              [model.cost, model.predictions], feed_dict={X: images_adv_curr, Y: curr_labels})

            (summaries, loss, predictions, truth, train_step) = sess.run(
              [model.summaries, model.cost, model.predictions,
               model.labels, model.global_step], feed_dict={X: curr_batch, Y: curr_labels})

            truth = np.argmax(truth, axis=1)
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
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f, adv: %.3f, best adv: %.3f, ' %
                        (loss, precision, best_precision, adv_precision, best_precision_adv))
        summary_writer.flush()

        # Storing adversarial samples in external array
        if adv_exist_flag == 0:
            np.save('adv_samples/'+args.target_model+'_{}.npy'.format(eps_num),images_adv)

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
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

    evaluate(hps, batch_size)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["fgs"])
    parser.add_argument('target_model', type=str,
                        help='name of target model')
    parser.add_argument("--src_model", type=str, default=None,
                        help="source model for attack")
    parser.add_argument("--eps", type=int, default=8,
                        help="FGS attack scale")

    args = parser.parse_args()
    log_root = 'logs_'+args.target_model
    eval_dir = log_root+'/eval'

    main()
