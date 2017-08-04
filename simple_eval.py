import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from fgs import symbolic_fgs, iter_fgs, symbolic_fg, symbolic_second_ord
from carlini_li import CarliniLi
from carlini_li_grad_free import CarliniLi_grad_free
from attack_utils import gen_grad, gen_hessian
from tf_utils import tf_test_error_rate, batch_eval
from os.path import basename
from time import time

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def main(attack, src_model_name, target_model_names):
    script_dir = os.path.dirname(__file__)
    np.random.seed(0)
    tf.set_random_seed(0)

    flags.DEFINE_integer('BATCH_SIZE', 1, 'Size of batches')
    set_mnist_flags()

    dim = FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS*FLAGS.NUM_CHANNELS

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = data_mnist()

    # source model for crafting adversarial examples
    src_model = load_model(src_model_name)

    # model(s) to target
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        target_models[i] = load_model(target_model_names[i])

    # simply compute test error
    if attack == "test":
        _,_, err = tf_test_error_rate(src_model, x, X_test, Y_test)
        print '{}: {:.1f}'.format(basename(src_model_name), err)

        for (name, target_model) in zip(target_model_names, target_models):
            err = tf_test_error_rate(target_model, x, X_test, Y_test)
            print '{}: {:.1f}'.format(basename(name), err)
        return

    eps = args.eps

    # take the random step in the RAND+FGSM
    if attack == "rand_fgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = src_model(x)
    grad = gen_grad(x, logits, y)

    # FGSM and RAND+FGSM one-shot attack
    if attack in ["fgs", "rand_fgs"] and args.norm == 'inf':
        adv_x = symbolic_fgs(x, grad, eps=eps)
    elif attack in ["fgs", "rand_fgs"] and args.norm == 'two':
        adv_x = symbolic_fg(x, grad, eps=eps)

    if attack == "second_ord":
        # x_rs = tf.reshape(x, [dim])
        x_rs = K.placeholder((dim))
        y_rs = K.placeholder((FLAGS.NUM_CLASSES))
        logits_rs = src_model(tf.reshape(x_rs, (1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)))
        grad_rs = gen_grad(x_rs, logits_rs, y_rs)
        # hessian = gen_hessian(x_rs, logits_rs, y_rs)
        # adv_x = symbolic_second_ord(x, grad_rs, hessian, dim, eps=eps)
        adv_x = symbolic_fgs(x_rs, grad_rs, eps=eps)
        X_test = X_test.reshape((10000, dim))
        l=1
        # X_adv = batch_eval([x_rs, y_rs], [adv_x], [X_test[0:l], Y_test[0:l]])[0]
        X_adv = []
        for i in range(l):
            X_adv.append(K.get_session().run([adv_x], feed_dict={x_rs: X_test[i], y_rs: Y_test[i],
                                                        K.learning_phase(): 0})[0])
        X_adv = np.array(X_adv)
        X_adv = X_adv.reshape((l, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))

    # iterative FGSM
    if attack == "ifgs":
        adv_x = iter_fgs(src_model, x, y, steps=args.steps, alpha = 0.01, eps=args.eps)

    # Carlini & Wagner attack
    if attack == "CW" or attack == "CW_gf":
        l = 1000
        pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_adv_'+str(args.eps)+'.p'
        print(pickle_name)
        Y_test = Y_test[0:l]
        if os.path.exists(pickle_name) and attack == "CW":
            print 'Loading adversarial samples'
            X_adv = pickle.load(open(pickle_name,'rb'))
            ofile = open('output_data/'+attack+'_attack_success.txt','a')

            preds_adv, _, err = tf_test_error_rate(src_model, x, X_adv, Y_test)

            pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_labels_'+str(args.eps)+'.p'
            pickle.dump(preds_adv, open(pickle_name, 'wb'))

            print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(src_model_name), err)
            ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(src_model_name), err))
            for (name, target_model) in zip(target_model_names, target_models):
                    preds_adv,_,err = tf_test_error_rate(target_model, x, X_adv, Y_test)
                    print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(name), err)
                    ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(name), err))

            ofile.close()
            return
        X_test = X_test[0:l]
        time1 = time()
        if attack == "CW_gf":
            cli = CarliniLi_grad_free(K.get_session(), src_model, targeted=False,
                                    confidence=args.kappa, eps=args.eps, mu=args.mu)
        elif attack == "CW":
            cli = CarliniLi(K.get_session(), src_model, targeted=False,
                                    confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_test, -args.eps, args.eps)
        X_adv = X_test + r
        time2 = time()
        print("Run with Adam took {}s".format(time2-time1))

        pickle.dump(X_adv, open(pickle_name,'wb'))

        ofile = open('output_data/'+attack+'_attack_success.txt','a')

        preds, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test)
        print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(src_model_name), err)
        ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(src_model_name), err))
        for (name, target_model) in zip(target_model_names, target_models):
            pres, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(name), err)
            ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(name), err))

        ofile.close()
        return

    # compute the adversarial examples and evaluate
    if attack != "second_ord":
        X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]

    pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_adv_'+str(args.eps)+'.p'
    pickle.dump(X_adv, open(pickle_name,'wb'))
    pickle_name = 'orig_images.p'
    pickle.dump(X_test, open(pickle_name, 'wb'))

    # white-box attack
    l = len(X_adv)

    preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
    pickle_name = 'orig_labels.p'
    pickle.dump(orig, open(pickle_name, 'wb'))
    pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_labels_'+str(args.eps)+'.p'
    pickle.dump(preds_adv, open(pickle_name, 'wb'))

    preds_orig, _, _ = tf_test_error_rate(src_model,x, X_test, Y_test[0:l])
    pickle_name = basename(src_model_name)+'_labels.p'
    pickle.dump(preds_orig, open(pickle_name, 'wb'))

    print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(src_model_name), err)

    # black-box attack
    for (name, target_model) in zip(target_model_names, target_models):
        preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print '{}->{}: {:.1f}'.format(basename(src_model_name), basename(name), err)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW", "CW_gf", "second_ord"])
    parser.add_argument("src_model", help="source model for attack")
    parser.add_argument('target_models', nargs='*',
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=10,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='inf',
                        help="Norm to use for attack")
    parser.add_argument("--mu", type=float, default=1e-5,
                        help="finite difference scale")

    args = parser.parse_args()
    main(args.attack, args.src_model, args.target_models)
