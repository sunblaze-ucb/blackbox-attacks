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

def gen_grad_cw(x, logits, y):
    real = tf.reduce_sum(y*logits[0], 1)
    other = tf.reduce_max((1-y)*logits[0] - (y*10000), 1)
    loss = tf.maximum(0.0,real-other+args.kappa)
    if len(logits) >= 1:
        for i in range(1, len(logits)):
            real = tf.reduce_sum(y*logits[i], 1)
            other = tf.reduce_max((1-y)*logits[i] - (y*10000), 1)
            loss += tf.maximum(0.0,real-other+args.kappa)
    grad = -1.0 * K.gradients(loss, [x])[0]
    return grad


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
        grad = gen_grad_cw(x, logits, y)

    ofile = open('output_data/blind_transfer/'+src_model_name_joint+'_to_'+basename(target_model_name)+'_'+args.loss_type+'.txt', 'a')

    if args.norm == 'linf':
        eps_list = list(np.linspace(0.025, 0.1, 4))
        eps_list.extend(np.linspace(0.15, 0.5, 8))
        if attack == "ifgs":
            eps_list = [0.3]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 2.0, 5))
        eps_list.extend(np.linspace(2.5, 9.0, 14))
        # eps_list = [5.0]
    print(eps_list)


    for eps in eps_list:
        # FGSM and RAND+FGSM one-shot attack
        if attack in ["fgs", "rand_fgs"] and args.norm == 'linf':
            adv_x = symbolic_fgs(x, grad, eps=eps)
        elif attack in ["fgs", "rand_fgs"] and args.norm == 'l2':
            adv_x = symbolic_fg(x, grad, eps=eps)

        # iterative FGSM
        if attack == "ifgs":
            pickle_name = 'adv_samples/' + attack + '/' + src_model_name_joint+'_'+'_'+args.loss_type+'_'+str(eps)+'_adv.p'
            if os.path.exists(pickle_name):
                X_adv = pickle.load(open(pickle_name,'rb'))
                avg_l2_perturb = np.mean(np.linalg.norm((X_adv-X_test).reshape(len(X_test),dim),axis=1))
                l = len(X_adv)

                for (name, src_model) in zip(src_model_names, src_models):
                    preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
                    print '{}->{}: {:.1f}'.format(basename(name), basename(name), err)

                # black-box attack
                preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
                print '{}->{}: {:.1f}, {}, {} {}'.format(src_model_name_joint, basename(target_model_name), err, avg_l2_perturb, eps, attack)
                ofile.write('{} {:.2f} {} \n'.format(eps, err, avg_l2_perturb))
                return


            adv_x = x
            # iteratively apply the FGSM with small step size
            for i in range(args.num_iter):
                adv_logits = [None] * len(src_model_names)
                for i in range(len(src_model_names)):
                    curr_model = src_models[i]
                    adv_logits[i] = curr_model(adv_x)

                if args.loss_type == 'xent':
                    loss, grad = gen_grad_ens(adv_x, adv_logits, y)
                elif args.loss_type == 'cw':
                    grad = gen_grad_cw(adv_x, adv_logits, y)

                adv_x = symbolic_fgs(adv_x, grad, args.delta, True)
                r = adv_x - x
                r = K.clip(r, -eps, eps)
                adv_x = x+r

            adv_x = K.clip(adv_x, 0, 1)

        if attack == "CW_ens":
            l = 1000
            pickle_name = 'adv_samples/' + attack + '/' + src_model_name_joint+'_'+str(args.eps)+'_adv.p'
            print(pickle_name)
            Y_test = Y_test[0:l]
            if os.path.exists(pickle_name) and attack == "CW_ens":
                print 'Loading adversarial samples'
                X_adv = pickle.load(open(pickle_name,'rb'))
                ofile = open('output_data/'+attack+'_attack_success.txt','a')

                for (name, src_model) in zip(src_model_names, src_models):
                    preds_adv, _, err = tf_test_error_rate(src_model, x, X_adv, Y_test)
                    print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(name), err)
                    ofile.write('{}->{}: {:.1f} \n'.format(src_model_name_joint, basename(name), err))
                    # pickle_name = attack + '_adv_samples/' + basename(src_model_name)+'_labels_'+str(args.eps)+'.p'
                    # pickle.dump(preds_adv, open(pickle_name, 'wb'))
                preds_adv,_,err = tf_test_error_rate(target_model, x, X_adv, Y_test)
                print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(target_model_name), err)
                ofile.write('{}->{}: {:.1f} \n'.format(src_model_name_joint, basename(target_model_name), err))

                ofile.close()
                return

            X_test = X_test[0:l]
            time1 = time()
            cli = CarliniLiEns(K.get_session(), src_models, targeted=False,
                                        confidence=args.kappa, eps=eps)

            X_adv = cli.attack(X_test, Y_test)

            r = np.clip(X_adv - X_test, -eps, eps)
            X_adv = X_test + r
            time2 = time()
            print("Run with Adam took {}s".format(time2-time1))

            pickle.dump(X_adv, open(pickle_name,'wb'))

            ofile = open('output_data/'+attack+'_attack_success.txt','a')

            preds, orig, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(target_model_name), err)
            ofile.write('{}->{}: {:.1f} \n'.format(src_model_name_joint, basename(target_model_name), err))
            for (name, src_model) in zip(src_model_names, src_models):
                pres, _, err = tf_test_error_rate(src_model, x, X_adv, Y_test)
                print '{}->{}: {:.1f}'.format(src_model_name_joint, basename(name), err)
                ofile.write('{}->{}: {:.1f} \n'.format(src_model_name_joint, basename(name), err))

            ofile.close()
            return

        X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
        avg_l2_perturb = np.mean(np.linalg.norm((X_adv-X_test).reshape(len(X_test),dim),axis=1))

        pickle_name = 'adv_samples/' + attack + '/' + src_model_name_joint+'_'+'_'+args.loss_type+'_'+str(eps)+'_adv.p'
        pickle.dump(X_adv, open(pickle_name,'wb'))

        # white-box attack
        l = len(X_adv)

        for (name, src_model) in zip(src_model_names, src_models):
            preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
            print '{}->{}: {:.1f}'.format(basename(name), basename(name), err)

        # black-box attack
        preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
        print '{}->{}: {:.1f}, {}, {} {}'.format(src_model_name_joint, basename(target_model_name), err, avg_l2_perturb, eps, attack)
        ofile.write('{} {:.2f} {} \n'.format(eps, err, avg_l2_perturb))
    ofile.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW_ens"])
    parser.add_argument('src_models', nargs='*',
                        help="source models for attack")
    parser.add_argument('--target_model',type=str,
                        help='path to target model(s)')
    # parser.add_argument("--eps", type=float, default=0.3,
    #                     help="FGS attack scale")
    parser.add_argument("--loss_type", type=str, default='xent',
                        help="Type of loss to use")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Iterated FGS step size")
    parser.add_argument("--num_iter", type=int, default=40,
                        help="Iterated FGS step size")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='linf',
                        help="Norm to use for attack")

    args = parser.parse_args()
    main(args.attack, args.src_models, args.target_model)
