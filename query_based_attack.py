import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from tf_utils import tf_test_error_rate, batch_eval

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

TRIAL_NUM = 100

def main(target_model_name):
    np.random.seed(0)
    tf.set_random_seed(0)

    set_mnist_flags()
    clip_min = 0
    clip_max = 1

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    dim = int(FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS)

    _, _, X_test, Y_test = data_mnist()
    print('Loaded data')

    # target model for crafting adversarial examples
    target_model = load_model(target_model_name)

    logits = target_model(x)
    prediction = K.softmax(logits)

    sess = tf.Session()
    print('Creating session')


    targets = np.random.randint(10, size = 100)

    success = 0
    for i in range(100):
        curr_sample = X_test[i].reshape((1, 28, 28 , 1))
        curr_label = Y_test[i].reshape((1,10))
        curr_target = targets[i]
        curr_prediction = K.get_session().run([prediction], feed_dict={x: curr_sample, K.learning_phase(): 0})[0]
        if np.argmax(curr_prediction, 1) != np.argmax(curr_label, 1):
            continue
        p_t = curr_prediction[:, curr_target]
        grad_est = np.zeros((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
        for j in range(dim):
            basis_vec = np.zeros((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
            row = int(j/FLAGS.IMAGE_COLS)
            col = int(j % FLAGS.IMAGE_COLS)
            basis_vec[:, row, col] = 1.
            x_plus_i = np.clip(curr_sample + args.delta * basis_vec, clip_min, clip_max)
            x_minus_i = np.clip(curr_sample - args.delta * basis_vec, clip_min, clip_max)
            pred_plus_t = K.get_session().run([prediction], feed_dict={x: x_plus_i, K.learning_phase(): 0})[0][:, curr_target]
            pred_minus_t = K.get_session().run([prediction], feed_dict={x: x_minus_i, K.learning_phase(): 0})[0][:, curr_target]
            grad_est[0, row, col] = (pred_plus_t - pred_minus_t)/args.delta
        x_adv = curr_sample + args.eps * (1/p_t) * grad_est
        adv_prediction = K.get_session().run([prediction], feed_dict={x: x_adv, K.learning_phase(): 0})[0]
        if np.argmax(adv_prediction, 1) == curr_target:
            success += 1
            print('{}'.format(success))
        # success = np.sum(np.argmax(adv_prediction,1) == targets)
    print('{}'.format(success))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="local perturbation")
    parser.add_argument("--norm", type=str, default='inf',
                            help="Norm to use for attack")
    args = parser.parse_args()

    main(args.target_model)
