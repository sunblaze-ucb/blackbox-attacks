import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from tf_utils import tf_test_error_rate, batch_eval
from keras.utils import np_utils
from attack_utils import gen_grad
from matplotlib import image as img
import time
from os.path import basename
from pyswarm import pso
import argparse

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

BATCH_SIZE = 100
BATCH_EVAL_NUM = 100

# PSO parameters
swarmsize = 100
maxiter = 100
omega = 0.5
p_wt = 0.3
s_wt = 0.7


parser = argparse.ArgumentParser()
parser.add_argument("target_model", help="target model for attack")

args = parser.parse_args()

eps = 0.3

target_model_name = basename(args.target_model)

set_mnist_flags()


np.random.seed(0)
tf.set_random_seed(0)

x = K.placeholder((None,
                   FLAGS.IMAGE_ROWS,
                   FLAGS.IMAGE_COLS,
                   FLAGS.NUM_CHANNELS))

y = K.placeholder((None, FLAGS.NUM_CLASSES))

dim = int(FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS)

_, _, X_test, Y_test = data_mnist()
print('Loaded data')

# target model for crafting adversarial examples
target_model = load_model(args.target_model)
target_model_name = basename(target_model_name)

logits = target_model(x)
prediction = K.softmax(logits)

sess = tf.Session()
print('Creating session')

targets = np.argmax(Y_test[:BATCH_SIZE*BATCH_EVAL_NUM], 1)
# elif RANDOM is False:
#     targets = np.array([target]*(BATCH_SIZE*BATCH_EVAL_NUM))
# elif RANDOM is True:
#     targets = np.random.randint(10, size = BATCH_SIZE*BATCH_EVAL_NUM)
targets_cat = np_utils.to_categorical(targets, FLAGS.NUM_CLASSES).astype(np.float32)

def loss(X):
    X = X.reshape((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    confidence = K.get_session().run([prediction], feed_dict={x: X, K.learning_phase(): 0})[0]
    # confidence[:,curr_target] = 1e-4
    max_conf_i = np.argmax(confidence, 1)
    max_conf = np.max(confidence, 1)[0]
    if max_conf_i == curr_target:
        return max_conf
    elif max_conf_i != curr_target:
        return -1.0 * max_conf

success = 0
adv_conf_avg = 0.0
sample_num = 1000

ofile = open('output_data/pso_adv_success.txt', 'a')

time1 = time.time()
for i in range(sample_num):
    print(i)
    X_ini = X_test[i].reshape(dim)
    curr_target = targets[i]

    ones_vec = np.ones_like(X_ini)

    lower_bound = np.clip(X_ini - eps * ones_vec, 0, 1)
    upper_bound = np.clip(X_ini + eps * ones_vec, 0 ,1)

    Xopt, fopt = pso(loss, lower_bound, upper_bound, swarmsize = 100, maxiter=100, debug = False)

    Xopt = Xopt.reshape((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    adv_pred_np = K.get_session().run([prediction], feed_dict={x: Xopt, K.learning_phase(): 0})[0]
    adv_label = np.argmax(adv_pred_np, 1)
    adv_conf = np.max(adv_pred_np, 1)
    if adv_label[0]!=curr_target:
        success += 1
        adv_conf_avg += adv_conf[0]

    if i<5:
        img.imsave('images/pso/'+
                    '{}_{}_{}_{}_{}.png'.format(target_model_name, i,
                    adv_label, curr_target, eps),
        Xopt.reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)*255, cmap='gray')

    # print(fopt, adv_conf[0])
time2 = time.time()
ofile.write('PSO params: swarmsize {}, maxiter {}, omega {}, p_wt {}, s_wt {} \n'
.format(swarmsize, maxiter, omega, p_wt, s_wt))
adv_conf_avg = adv_conf_avg/success
ofile.write('{}, {}: {} of {}, {} \n'.format(target_model_name, eps, success, sample_num, adv_conf_avg))
print(success)
print('{:.2f}'.format((time2-time1)/sample_num))
