import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from fgs import symbolic_fgs, iter_fgs, symbolic_fg
from carlini_li_ens import CarliniLiEns
from attack_utils import gen_grad, gen_grad_ens
from tf_utils import tf_test_error_rate, batch_eval
from os.path import basename
from time import time

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def main(attack, src_model_names, target_model_name):
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
    src_models = [None] * len(src_model_names)
    for i in range(len(src_model_names)):
        src_models[i] = load_model(src_model_names[i])

    src_model_name_joint = ''
    for i in range(len(src_models)):
        src_model_name_joint += basename(src_model_names[i])

    # model(s) to target
    target_model = load_model(target_model_name)

    # simply compute test error
    if attack == "test":
        for (name, src_model) in zip(src_model_names, src_models):
            _, _, err = tf_test_error_rate(src_model, x, X_test, Y_test)
            print '{}: {:.1f}'.format(basename(name), err)

        _,_,err = tf_test_error_rate(target_model, x, X_test, Y_test)
        print '{}: {:.1f}'.format(basename(target_model_name), err)

        return

    eps = args.eps

    # take the random step in the RAND+FGSM
    if attack == "rand_fgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = [None] * len(src_model_names)
    for i in range(len(src_model_names)):
        curr_model = src_models[i]
        logits[i] = curr_model(x)

    if args.loss_type == 'xent':
        loss, grad = gen_grad_ens(x, logits, y)
    elif args.loss_type == 'cw':
        real = tf.reduce_sum(y*logits[0], 1)
        other = tf.reduce_max((1-y)*logits[0] - (y*10000), 1)
        loss = tf.maximum(0.0,real-other+args.kappa)
        if len(logits) >= 1:
            for i in range(1, len(logits)):
                real = tf.reduce_sum(y*logits[i], 1)
                other = tf.reduce_max((1-y)*logits[i] - (y*10000), 1)
                loss += tf.maximum(0.0,real-other+args.kappa)
        grad = -1.0 * K.gradients(loss, [x])[0]

    # FGSM and RAND+FGSM one-shot attack
    if attack in ["fgs", "rand_fgs"] and args.norm == 'inf':
        adv_x = symbolic_fgs(x, grad, eps=eps)
    elif attack in ["fgs", "rand_fgs"] and args.norm == 'two':
        adv_x = symbolic_fg(x, grad, eps=eps)

    if attack == "CW_ens":
        l = 10
        pickle_name = attack + '_adv_samples/' + src_model_name_joint+'_adv_'+str(args.eps)+'.p'
        print(pickle_name)
        Y_test = Y_test[0:l]
        if os.path.exists(pickle_name) and attack == "CW":
            print 'Loading adversarial samples'
            X_adv = pickle.load(open(pickle_name,'rb'))
            ofile = open('output_data/'+attack+'_attack_success.txt','a')

            for (name, src_model) in zip(src_model_names, src_models):
                preds_adv, _, err = tf_test_error_rate(src_model, x, X_adv, Y_test)

            # pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_labels_'+str(args.eps)+'.p'
            # pickle.dump(preds_adv, open(pickle_name, 'wb'))

                print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(name), err)
                # ofile.write('{}->{}: {:.1f} \n'.format(src_model_name_joint, basename(src_model_name), err))
            preds_adv,_,err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(target_model_name), err)
            # ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(name), err))

            ofile.close()
            return

        X_test = X_test[0:l]
        time1 = time()
        cli = CarliniLiEns(K.get_session(), src_models, targeted=False,
                                    confidence=args.kappa, eps=args.eps)

        X_adv = cli.attack(X_test, Y_test)

        r = np.clip(X_adv - X_test, -args.eps, args.eps)
        X_adv = X_test + r
        time2 = time()
        print("Run with Adam took {}s".format(time2-time1))

        # pickle.dump(X_adv, open(pickle_name,'wb'))

        # ofile = open('output_data/'+attack+'_attack_success.txt','a')

        preds, orig, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(target_model_name), err)
        # ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(src_model_name), err))
        for (name, src_model) in zip(src_model_names, src_models):
            pres, _, err = tf_test_error_rate(src_model, x, X_adv, Y_test)
            print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(name), err)
            # ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(name), err))

        # ofile.close()
        return


    X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]

    # pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_adv_'+str(args.eps)+'.p'
    # pickle.dump(X_adv, open(pickle_name,'wb'))
    # pickle_name = 'orig_images.p'
    # pickle.dump(X_test, open(pickle_name, 'wb'))

    # white-box attack
    l = len(X_adv)

    for (name, src_model) in zip(src_model_names, src_models):
        preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
        print '{}->{}: {:.1f}'.format(basename(name), basename(name), err)
    # pickle_name = 'orig_labels.p'
    # pickle.dump(orig, open(pickle_name, 'wb'))
    # pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_labels_'+str(args.eps)+'.p'
    # pickle.dump(preds_adv, open(pickle_name, 'wb'))

    # preds_orig, _, _ = tf_test_error_rate(src_model,x, X_test, Y_test[0:l])
    # pickle_name = basename(src_model_name)+'_labels.p'
    # pickle.dump(preds_orig, open(pickle_name, 'wb'))



    # black-box attack
    preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
    print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(target_model_name), err)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW_ens"])
    parser.add_argument('src_models', nargs='*',
                        help="source models for attack")
    parser.add_argument('--target_model',type=str,
                        help='path to target model(s)')
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--loss_type", type=str, default='xent',
                        help="Type of loss to use")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=10,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='inf',
                        help="Norm to use for attack")
    parser.add_argument("--mu", type=float, default=1e-5,
                        help="finite difference scale")

    args = parser.parse_args()
    main(args.attack, args.src_models, args.target_model)
