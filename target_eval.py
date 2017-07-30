import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
import os
from mnist import data_mnist, set_mnist_flags, load_model
from fgs import symbolic_fgs, iter_fgs, symbolic_fg
from carlini_li import CarliniLi
from attack_utils import gen_grad
from tf_utils import tf_test_error_rate, batch_eval
from os.path import basename
from matplotlib import image as img

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def main(attack, target_model_name, source_model_names):
    script_dir = os.path.dirname(__file__)
    np.random.seed(0)
    tf.set_random_seed(0)

    flags.DEFINE_integer('BATCH_SIZE', 10, 'Size of batches')
    flags.DEFINE_integer('IMAGE_NUM', 0, 'Number of images to print')
    set_mnist_flags()

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = data_mnist()

    # source model for crafting adversarial examples
    source_model_names.insert(0, target_model_name)
    source_models = [None] * (len(source_model_names))
    for i in range(len(source_models)):
	    source_models[i] = load_model(source_model_names[i])

    # model(s) to target
    target_model = load_model(target_model_name)

    print(target_model.summary())

    # simply compute test error
    if attack == "test":
        err = tf_test_error_rate(target_model, x, X_test, Y_test)
        print '{}: {:.1f}'.format(basename(target_model_name), err)

        for (name, src_model) in zip(source_model_names, source_models):
            err = tf_test_error_rate(src_model, x, X_test, Y_test)
            print '{}: {:.1f}'.format(basename(name), err)
        return

    eps_list = list(np.linspace(0.0, 0.1, 5))
    eps_list.extend(np.linspace(0.2, 0.5, 7))

    if args.eps is not None:
        eps_list = [args.eps]

    print(eps_list)

    for i in range(1,len(source_models)):
        src_model = source_models[i]
        src_model_name = source_model_names[i]
        logits = src_model(x)
        grad = gen_grad(x, logits, y)

        ofile = open('output_data/blind_transfer/'+basename(src_model_name)+'_to_'+basename(target_model_name)+'.txt', 'a')
        # ofile.write(args.attack+'\n')
        for eps in eps_list:
            # take the random step in the RAND+FGSM
            if attack == "rand_fgs":
                X_test = np.clip(
                    X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
                    0.0, 1.0)
                eps -= args.alpha

            # FGSM and RAND+FGSM one-shot attack
            if attack in ["fgs", "rand_fgs"] and args.norm == 'inf':
                adv_x = symbolic_fgs(x, grad, eps=eps)
            elif attack in ["fgs", "rand_fgs"] and args.norm == 'two':
                adv_x = symbolic_fg(x, grad, eps=eps)

            # iterative FGSM
            if attack == "ifgs":
                adv_x = iter_fgs(src_model, x, y, steps=args.steps, eps=eps/args.steps)

            # Carlini & Wagner attack
            if attack == "CW":
                ofile = open('output_data/CW_attack_success.txt','a')
                l = 1000
                pickle_name = 'CW_adv_samples/' + basename(src_model_name) +'_adv_'+str(eps)+'.p'
                Y_test = Y_test[0:l]
                if os.path.exists(pickle_name):
                    print 'Loading adversarial samples'
                    X_adv = pickle.load(open(pickle_name,'rb'))

                    #err = tf_test_error_rate(src_model, x, X_adv, Y_test)
                    #print '{}->{}: {:.1f}, {} {}'.format(basename(src_model_name), basename(src_model_name), err, eps, attack)
                    #ofile.write('{}->{}: {:.1f}, {} \n'.format(basename(src_model_name), basename(src_model_name), err, eps, attack))
                    _, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
                    print '{}->{}: {:.1f}, {}'.format(basename(src_model_name), basename(target_model_name), err, eps, attack)
                    print '{}->{}: {:.1f}, {}'.format(basename(src_model_name), basename(target_model_name), err, eps, attack)
                    ofile.write('{}->{}: {:.1f} \n'.format(basename(src_model_name), basename(target_model_name), err, eps, attack))
                    continue

                X_test = X_test[0:l]

                cli = CarliniLi(K.get_session(), src_model,
                                targeted=False, confidence=args.kappa, eps=eps)

                X_adv = cli.attack(X_test, Y_test)

                r = np.clip(X_adv - X_test, -args.eps, args.eps)
                X_adv = X_test + r
                pickle.dump(X_adv, open(pickle_name,'wb'))

               # err = tf_test_error_rate(src_model, x, X_adv, Y_test)
               # print '{}->{}: {:.1f}, {} {}'.format(basename(src_model_name), basename(src_model_name), err, eps, attack)
               # ofile.write('{}->{}: {:.1f}, {} \n'.format(basename(src_model_name), basename(src_model_name), err, eps, attack))
                _, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
                print '{}->{}: {:.1f}, {}'.format(basename(src_model_name), basename(target_model_name), err, eps, attack)
                ofile.write('{} {:.2f} \n'.format(basename(src_model_name), basename(target_model_name), err, eps, attack))

                continue

            # compute the adversarial examples and evaluate
            X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
            # print(X_adv.shape())

            rel_path_i = 'images/'
            abs_path_i = os.path.join(script_dir, rel_path_i)
            for i in range(FLAGS.IMAGE_NUM):
                img.imsave(abs_path_i + basename(src_model_name) + '_{}_mag{}.png'.format(i, eps),
                    X_adv[i].reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS) * 255,
                            vmin=0, vmax=255, cmap='gray')

            # first run is white-box, then black-box attacks
            _, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            print '{}->{}: {:.1f}, {} {}'.format(basename(src_model_name), basename(target_model_name), err, eps, attack)
            ofile.write('{} {:.2f} \n'.format(eps, err))
    ofile.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW"])
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument('source_models', nargs='*',
                            help='path to source model(s)')
    parser.add_argument("--eps", type=float, default=None,
                       help="FGS attack scale")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--steps", type=int, default=10,
                        help="Iterated FGS steps")
    parser.add_argument("--kappa", type=float, default=100,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='inf',
                        help="Norm to use for attack")

    args = parser.parse_args()
    main(args.attack, args.target_model, args.source_models)
