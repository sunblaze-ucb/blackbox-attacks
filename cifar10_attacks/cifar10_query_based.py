import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
import models
from tf_utils import tf_test_error_rate, batch_eval
from keras.utils import np_utils
from attack_utils import gen_grad
import time
from os.path import basename

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

NUM_CLASSES = 10
IMAGE_ROWS = 32
IMAGE_COLS = 32
NUM_CHANNELS = 3

RANDOM = True
BATCH_SIZE = 100
CLIP_MIN = 0.0
CLIP_MAX = 255.0

def wb_write_out(eps, white_box_success, wb_norm):
    if RANDOM is False:
        print('Fraction of targets achieved (white-box) for {}: {}'.format(target, white_box_success))
    else:
        print('Fraction of targets achieved (white-box): {}'.format(white_box_success))
    return

def est_write_out(eps, success, avg_l2_perturb):
    if RANDOM is False:
        print('Fraction of targets achieved (query-based) with {} for {}: {}'.format(target_model_name, target, success))
    else:
        print('Fraction of targets achieved (query-based): {}'.format(success))
    return

def pca_components(X, dim):
    X = X.reshape((len(X), dim))
    pca = PCA(n_components=dim)
    pca.fit(X)

    U = (pca.components_).T
    U_norm = normalize(U, axis=0)

    return U_norm[:,:args.num_comp]


def xent_est(prediction, x, x_plus_i, x_minus_i, curr_target):
    pred_plus = sess.run(prediction, feed_dict={x: x_plus_i})
    pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]
    pred_minus = sess.run(prediction, feed_dict={x: x_minus_i})
    pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]
    single_grad_est = (pred_plus_t - pred_minus_t)/args.delta

    return single_grad_est/2.0

def CW_est(logits, x, x_plus_i, x_minus_i, curr_sample, curr_target):
    curr_logits = sess.run(logits, feed_dict={x: curr_sample})
    # So that when max is taken, it returns max among classes apart from the
    # target
    curr_logits[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
    max_indices = np.argmax(curr_logits, 1)
    logit_plus = sess.run(logits, feed_dict={x: x_plus_i})
    logit_plus_t = logit_plus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_plus_max = logit_plus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_minus = sess.run(logits, feed_dict={x: x_minus_i})
    logit_minus_t = logit_minus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_minus_max = logit_minus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_t_grad_est = (logit_plus_t - logit_minus_t)/args.delta
    logit_max_grad_est = (logit_plus_max - logit_minus_max)/args.delta

    return logit_t_grad_est/2.0, logit_max_grad_est/2.0


def finite_diff_method(prediction, logits, x, curr_sample, curr_target, p_t, dim, U=None):
    grad_est = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    logits_np = sess.run(logits, feed_dict={x: curr_sample})
    if PCA_FLAG == False:
        random_indices = np.random.permutation(dim)
        num_groups = dim / args.group_size
    elif PCA_FLAG == True:
        num_groups = args.num_comp
    for j in range(num_groups):
        if j%100==0:
            print('Group no.:{}'.format(j))
        basis_vec = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

        if PCA_FLAG == False:
            if j != num_groups-1:
                curr_indices = random_indices[j*args.group_size:(j+1)*args.group_size]
            elif j == num_groups-1:
                curr_indices = random_indices[j*args.group_size:]
            per_c_indices = curr_indices%(IMAGE_COLS*IMAGE_ROWS)
            channel = curr_indices/(IMAGE_COLS*IMAGE_ROWS)
            row = per_c_indices/IMAGE_COLS
            col = per_c_indices % IMAGE_COLS
            for i in range(len(curr_indices)):
                basis_vec[:, row[i], col[i], channel[i]] = 1.

        elif PCA_FLAG == True:
            basis_vec[:] = U[:,j].reshape((1, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

        x_plus_i = np.clip(curr_sample + args.delta * basis_vec, CLIP_MIN, CLIP_MAX)
        x_minus_i = np.clip(curr_sample - args.delta * basis_vec, CLIP_MIN, CLIP_MAX)

        if args.loss_type == 'cw':
            logit_t_grad_est, logit_max_grad_est = CW_est(logits, x, x_plus_i,
                                            x_minus_i, curr_sample, curr_target)
            if '_un' in args.method:
                single_grad_est = logit_t_grad_est - logit_max_grad_est
            else:
                single_grad_est = logit_max_grad_est - logit_t_grad_est
        elif args.loss_type == 'xent':
            single_grad_est = xent_est(prediction, x, x_plus_i, x_minus_i, curr_target)
        if PCA_FLAG == False:
            for i in range(len(curr_indices)):
                grad_est[:, row[i], col[i], channel[i]] = single_grad_est.reshape((BATCH_SIZE))
        elif PCA_FLAG == True:
            grad_est += basis_vec*single_grad_est[:,None,None,None]

    # Getting gradient of the loss
    if args.loss_type == 'xent':
        loss_grad = -1.0 * grad_est/p_t[:, None, None, None]
    elif args.loss_type == 'cw':
        logits_np_t = logits_np[np.arange(BATCH_SIZE), list(curr_target)].reshape(BATCH_SIZE)
        logits_np[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
        max_indices = np.argmax(logits_np, 1)
        logits_np_max = logits_np[np.arange(BATCH_SIZE), list(max_indices)].reshape(BATCH_SIZE)
        logit_diff = logits_np_t - logits_np_max
        if '_un' in args.method:
            zero_indices = np.where(logit_diff + args.conf < 0.0)
        else:
            zero_indices = np.where(-logit_diff + args.conf < 0.0)
        grad_est[zero_indices[0]] = np.zeros((len(zero_indices), IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        loss_grad = grad_est

    return loss_grad

def one_shot_method(prediction, x, curr_sample, curr_target, p_t):
    grad_est = np.zeros((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    DELTA = np.random.randint(2, size=(BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    np.place(DELTA, DELTA==0, -1)

    y_plus = np.clip(curr_sample + args.delta * DELTA, CLIP_MIN, CLIP_MAX)
    y_minus = np.clip(curr_sample - args.delta * DELTA, CLIP_MIN, CLIP_MAX)

    if args.CW_loss == 0:
        pred_plus = K.get_session().run([prediction], feed_dict={x: y_plus, K.learning_phase(): 0})[0]
        pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]

        pred_minus = K.get_session().run([prediction], feed_dict={x: y_minus, K.learning_phase(): 0})[0]
        pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]

        num_est = (pred_plus_t - pred_minus_t)

    grad_est = num_est[:, None, None, None]/(args.delta * DELTA)

    # Getting gradient of the loss
    if args.CW_loss == 0:
        loss_grad = -1.0 * grad_est/p_t[:, None, None, None]

    return loss_grad

def estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim):
    success = 0
    avg_l2_perturb = 0.0
    time1 = time.time()
    X_test_mod = X_test[random_indices]
    X_test_ini_mod = X_test_ini[random_indices]
    U = None
    if PCA_FLAG == True:
        U = pca_components(X_test, dim)
    for i in range(BATCH_EVAL_NUM):
        if i % 10 ==0:
            print('{}, {}'.format(i, eps))
        curr_sample = X_test_mod[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        curr_sample_ini = X_test_ini_mod[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

        curr_target = targets[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        curr_prediction = sess.run(prediction, feed_dict={x: curr_sample})

        p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

        if 'query_based' in args.method:
            loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                            curr_target, p_t, dim, U)
        elif 'one_shot' in args.method:
            loss_grad = one_shot_method(prediction, x, curr_sample, curr_target, p_t)

        # Getting signed gradient of loss
        if args.norm == 'linf':
            normed_loss_grad = np.sign(loss_grad)
        elif args.norm == 'l2':
            grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis = 1)
            indices = np.where(grad_norm != 0.0)
            normed_loss_grad = np.zeros_like(curr_sample)
            normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

        eps_mod = eps - args.alpha
        if args.loss_type == 'xent':
            if '_un' in args.method:
                x_adv = np.clip(curr_sample + eps_mod * normed_loss_grad, CLIP_MIN, CLIP_MAX)
            else:
                x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, CLIP_MIN, CLIP_MAX)
        elif args.loss_type == 'cw':
            x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, CLIP_MIN, CLIP_MAX)

        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv-curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = sess.run(prediction, feed_dict={x: x_adv})
        success += np.sum(np.argmax(adv_prediction,1) == curr_target)
        
    success = 100.0 * float(success)/(BATCH_SIZE*BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb/BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    return

def estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, beta):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    U = None
    X_test_mod = X_test[random_indices]
    X_test_ini_mod = X_test_ini[random_indices]
    if PCA_FLAG == True:
        U = pca_components(X_test, dim)
    for i in range(BATCH_EVAL_NUM):
        if i % 10 ==0:
            print('{}, {}'.format(i, eps))
        curr_sample = X_test_mod[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        curr_sample_ini = X_test_ini_mod[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))

        curr_target = targets[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        eps_mod = eps - args.alpha

        for i in range(args.num_iter):
            curr_prediction = sess.run(prediction, feed_dict={x: curr_sample})

            p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

            if 'query_based' in args.method:
                loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                            curr_target, p_t, dim, U)
            elif 'one_shot' in args.method:
                loss_grad = one_shot_method(prediction, x, curr_sample, curr_target, p_t)

            # Getting signed gradient of loss
            if args.norm == 'linf':
                normed_loss_grad = np.sign(loss_grad)
            elif args.norm == 'l2':
                grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis = 1)
                indices = np.where(grad_norm != 0.0)
                normed_loss_grad = np.zeros_like(curr_sample)
                normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

            if args.loss_type == 'xent':
                if '_un' in args.method:
                    x_adv = np.clip(curr_sample + beta * normed_loss_grad, CLIP_MIN, CLIP_MAX)
                else:
                    x_adv = np.clip(curr_sample - beta * normed_loss_grad, CLIP_MIN, CLIP_MAX)
            elif args.loss_type == 'cw':
                x_adv = np.clip(curr_sample - beta * normed_loss_grad, CLIP_MIN, CLIP_MAX)
            
            r = x_adv-curr_sample_ini
            r = np.clip(r, -eps, eps)
            curr_sample = curr_sample_ini + r

        x_adv = np.clip(curr_sample, CLIP_MIN, CLIP_MAX)
        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv-curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = sess.run(prediction, feed_dict={x: x_adv})
        success += np.sum(np.argmax(adv_prediction,1) == curr_target)

    success = 100.0 * float(success)/(BATCH_SIZE*BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb/BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2-time1, (time2 - time1)/(BATCH_SIZE*BATCH_EVAL_NUM)))

    return


def white_box_fgsm(prediction, target_model, x, logits, y, X_test, X_test_ini, targets, targets_cat, eps, dim):

    #Get gradient from model
    if args.loss_type == 'xent':
        grad = gen_grad(x, logits, y)
    elif args.loss_type == 'cw':
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if '_un' in args.method:
            loss = tf.maximum(0.0,real-other+args.conf)
        else:
            loss = tf.maximum(0.0,other-real+args.conf)
        grad, = tf.gradients(loss, x)

    # normalized gradient
    if args.norm == 'linf':
        normed_grad = tf.sign(grad)
    elif args.norm == 'l2':
        normed_grad = K.l2_normalize(grad, axis = (1,2,3))

    # Multiply by constant epsilon
    scaled_grad = (eps - args.alpha) * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if args.loss_type == 'xent':
        if '_un' in args.method:
            adv_x_t = tf.stop_gradient(x + scaled_grad)
        else:
            adv_x_t = tf.stop_gradient(x - scaled_grad)
    elif args.loss_type == 'cw':
        adv_x_t = tf.stop_gradient(x - scaled_grad)

    adv_x_t = tf.clip_by_value(adv_x_t, CLIP_MIN, CLIP_MAX)

    X_test_mod = X_test[random_indices]
    X_test_ini_mod = X_test_ini[random_indices]

    X_adv_t = np.zeros_like(X_test_mod)
    adv_pred_np = np.zeros((len(X_test_mod), NUM_CLASSES))
    pred_np = np.zeros((len(X_test_mod), NUM_CLASSES))

    for i in range(BATCH_EVAL_NUM):
        X_test_slice = X_test_mod[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        targets_cat_slice = targets_cat[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        x_adv_i = sess.run(adv_x_t, feed_dict={x: X_test_slice, y: targets_cat_slice})
        X_adv_t[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:,:,:] = x_adv_i
        adv_pred_np_i = sess.run(prediction, feed_dict={x: x_adv_i})
        adv_pred_np[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = adv_pred_np_i
        pred_np_i = sess.run(prediction, feed_dict={x: X_test_slice})
        pred_np[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = pred_np_i

    white_box_success = 100.0 * np.sum(np.argmax(adv_pred_np, 1) == targets)/len(X_test_mod)
    if '_un' in args.method:
        white_box_success = 100.0 - white_box_success
    benign_success = 100.0 * np.sum(np.argmax(pred_np, 1) == targets)/len(X_test_mod)
    print('Benign success: {}'.format(benign_success))

    wb_norm = np.mean(np.linalg.norm((X_adv_t-X_test_ini_mod).reshape(BATCH_SIZE*BATCH_EVAL_NUM, dim), axis=1))
    print('Average white-box l2 perturbation: {}'.format(wb_norm))

    wb_write_out(eps, white_box_success, wb_norm)

    return

def white_box_fgsm_iter(prediction, target_model, x, logits, y, X_test, X_test_ini, targets, targets_cat, eps, dim, beta):
    #Get gradient from model
    if args.loss_type == 'xent':
        grad = gen_grad(x, logits, y)
    elif args.loss_type == 'cw':
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if '_un' in args.method:
            loss = tf.maximum(0.0,real-other+args.conf)
        else:
            loss = tf.maximum(0.0,other-real+args.conf)
        grad, = tf.gradients(loss, x)

    # normalized gradient
    if args.norm == 'linf':
        normed_grad = tf.sign(grad)
    elif args.norm == 'l2':
        normed_grad = K.l2_normalize(grad, axis = (1,2,3))

    # Multiply by constant epsilon
    scaled_grad = beta * normed_grad

    # Add perturbation to original example to obtain adversarial example
    if args.loss_type == 'xent':
        if '_un' in args.method:
            adv_x_t = tf.stop_gradient(x + scaled_grad)
        else:
            adv_x_t = tf.stop_gradient(x - scaled_grad)
    elif args.loss_type == 'cw':
        adv_x_t = tf.stop_gradient(x - scaled_grad)

    adv_x_t = tf.clip_by_value(adv_x_t, CLIP_MIN, CLIP_MAX)

    X_test_mod = X_test[random_indices]
    X_test_ini_mod = X_test_ini[random_indices]

    X_adv_t = np.zeros_like(X_test_ini_mod)
    adv_pred_np = np.zeros((len(X_test_ini_mod), NUM_CLASSES))
    pred_np = np.zeros((len(X_test_ini_mod), NUM_CLASSES))

    for i in range(BATCH_EVAL_NUM):
        X_test_slice = X_test_mod[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        X_test_ini_slice = X_test_ini_mod[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        targets_cat_slice = targets_cat[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)]
        X_adv_curr = X_test_slice
        for k in range(args.num_iter):
            X_adv_curr = sess.run(adv_x_t, feed_dict={x: X_adv_curr, y: targets_cat_slice})
            r = X_adv_curr - X_test_ini_slice
            r = np.clip(r, -eps, eps)
            X_adv_curr = X_test_ini_slice + r
        X_adv_curr = np.clip(X_adv_curr, CLIP_MIN, CLIP_MAX)
        X_adv_t[i*(BATCH_SIZE):(i+1)*(BATCH_SIZE)] = X_adv_curr
        adv_pred_np_i = sess.run(prediction, feed_dict={x: X_adv_curr})
        adv_pred_np[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = adv_pred_np_i
        pred_np_i, logits_np_i = sess.run([prediction, logits], feed_dict={x: X_test_slice})
        pred_np[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:] = pred_np_i

    white_box_success = 100.0 * np.sum(np.argmax(adv_pred_np, 1) == targets)/(BATCH_SIZE*BATCH_EVAL_NUM)
    if '_un' in args.method:
        white_box_success = 100.0 - white_box_success
    benign_success = 100.0 * np.sum(np.argmax(pred_np, 1) == targets)/(BATCH_SIZE*BATCH_EVAL_NUM)
    print('Benign success: {}'.format(benign_success))

    wb_norm = np.mean(np.linalg.norm((X_adv_t-X_test_ini_mod).reshape(BATCH_SIZE*BATCH_EVAL_NUM, dim), axis=1))
    print('Average white-box l2 perturbation: {}'.format(wb_norm))

    wb_write_out(eps, white_box_success, wb_norm)

    return


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", help="target model for attack")
parser.add_argument("--img_source", help="source of images",
                    default='test_orig.npy')
parser.add_argument("--label_source", help="source of labels",
                    default='test_labels.npy')
parser.add_argument("--method", choices=['query_based', 'one_shot',
                    'query_based_un', 'one_shot_un','query_based_un_iter','query_based_iter'], default='query_based')
parser.add_argument("--delta", type=float, default=0.01,
                    help="local perturbation")
parser.add_argument("--norm", type=str, default='linf',
                        help="Norm to use for attack")
parser.add_argument("--loss_type", type=str, default='cw',
                        help="Choosing which type of loss to use")
parser.add_argument("--conf", type=float, default=0.0,
                            help="Strength of CW sample")
parser.add_argument("--alpha", type=float, default=0.0,
                        help="Strength of random perturbation")
parser.add_argument("--group_size", type=int, default=1,
                        help="Number of features to group together")
parser.add_argument("--num_comp", type=int, default=3072,
                        help="Number of pca components")
parser.add_argument("--num_iter", type=int, default=10,
                        help="Number of iterations")
parser.add_argument("--beta", type=int, default=1.0,
                        help="Step size per iteration")

args = parser.parse_args()

if '_un' in args.method:
    RANDOM = True
PCA_FLAG=False
if args.num_comp != 3072:
    PCA_FLAG=True

if '_iter' in args.method:
    BATCH_EVAL_NUM = 10
else:
    BATCH_EVAL_NUM = 10

target_model_name = args.ckpt_dir

np.random.seed(0)
tf.set_random_seed(0)

x = tf.placeholder(shape=(BATCH_SIZE,
                   IMAGE_ROWS,
                   IMAGE_COLS,
                   NUM_CHANNELS),dtype=tf.float32)

y = tf.placeholder(shape=(BATCH_SIZE, NUM_CLASSES),dtype=tf.float32)

dim = int(IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS)

X_test_ini = np.load(args.img_source)
Y_test = np.load(args.label_source)
print('Loaded data')

random_indices = np.random.choice(len(X_test_ini),BATCH_SIZE*BATCH_EVAL_NUM, replace=False)
Y_test = Y_test[random_indices]
Y_test_uncat = np.argmax(Y_test,1)

# target model for crafting adversarial examples
target_model = models.load_model('logs/'+target_model_name, BATCH_SIZE, x, y)

logits = target_model.get_logits()
prediction = tf.nn.softmax(logits)

sess = tf.Session()
target_model.load(sess)
print('Creating session')

if '_un' in args.method:
    targets = Y_test_uncat
elif RANDOM is False:
    targets = np.array([target]*(BATCH_SIZE*BATCH_EVAL_NUM))
elif RANDOM is True:
    targets = []
    allowed_targets = list(range(NUM_CLASSES))
    for i in range(BATCH_SIZE*BATCH_EVAL_NUM):
        allowed_targets.remove(Y_test_uncat[i])
        targets.append(np.random.choice(allowed_targets))
        allowed_targets = list(range(NUM_CLASSES))
    targets = np.array(targets)
targets_cat = np_utils.to_categorical(targets, NUM_CLASSES).astype(np.float32)

if args.norm == 'linf':
    # eps_list = list(np.linspace(4.0, 32.0, 8))
    eps_list = [8.0]
    if "_iter" in args.method:
        eps_list = [8.0]
        # eps_list = list(np.linspace(4.0, 32.0, 8))
elif args.norm == 'l2':
    eps_list = list(np.linspace(0.0, 2.0, 5))
    eps_list.extend(np.linspace(2.5, 9.0, 14))
    # eps_list = [5.0]
print eps_list

random_perturb = np.random.randn(*X_test_ini.shape)

if args.norm == 'linf':
    random_perturb_signed = np.sign(random_perturb)
    X_test = np.clip(X_test_ini + args.alpha * random_perturb_signed, CLIP_MIN, CLIP_MAX)
elif args.norm == 'l2':
    random_perturb_unit = random_perturb/np.linalg.norm(random_perturb.reshape(curr_len,dim), axis=1)[:, None, None, None]
    X_test = np.clip(X_test_ini + args.alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

for eps in eps_list:
    if '_iter' in args.method:
        white_box_fgsm_iter(prediction, target_model, x, logits, y, X_test, X_test_ini, targets, targets_cat, eps, dim, args.beta)
        estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, args.beta)
    else:
        white_box_fgsm(prediction, target_model, x, logits, y, X_test, X_test_ini, targets,
                    targets_cat, eps, dim)

        estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim)
