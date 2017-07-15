import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import data_mnist, set_mnist_flags, load_model
from os.path import basename
from matplotlib import image as img

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

CLIP_MIN = 0
CLIP_MAX = 1
C_TRIAL = True

def class_means(X, y):

    """Return a list of means of each class in (X,y)"""

    classes = np.unique(y)
    no_of_classes = len(classes)
    means = []
    class_frac = []
    for item in classes:
        indices = np.where(y == item)[0]
        class_items = X[indices]
        class_frac.append(float(len(class_items))/float(len(X)))
        mean = np.mean(class_items, axis=0)
        means.append(mean)
    return means, class_frac


def length_scales(X, y):

    """Find distances from each class mean to means of the other classes"""

    means, class_frac = class_means(X, y)
    no_of_classes = len(means)
    mean_dists = np.zeros((no_of_classes, no_of_classes))
    scales = []
    closest_means = np.zeros((no_of_classes))
    for i in range(no_of_classes):
        mean_diff = 0.0
        curr_mean = means[i]
        mean_not_i = 0.0
        curr_frac = class_frac[i]
        closest_mean = 1e6
        for j in range(no_of_classes):
            if i == j:
                mean_dists[i,j] = 0.0
            else:
                mean_dists[i,j] = np.linalg.norm(curr_mean-means[j])
                if mean_dists[i,j]<closest_mean:
                    closest_mean = mean_dists[i,j]
                    closest_means[i] = j
                mean_not_i = mean_not_i + means[j]

        mean_diff = curr_frac*curr_mean - (1-curr_frac)*(mean_not_i/(no_of_classes-1))
        scales.append(np.linalg.norm(mean_diff))
    return scales, mean_dists, closest_means



def naive_untargeted_attack(X, y):

    """
    Returns a minimum distance required to move a sample to a different class
    """

    scales = length_scales(X, y)
    print scales
    data_len = len(X)
    classes = np.unique(y)
    distances = []
    for i in range(100):
        curr_data = X[i,:]
        curr_distances = []
        for j in range(100):
            if i == j: continue
            else:
                # if y[i]==y[j]:
                #     continue
                if y[i] != y[j]:
                    data_diff = curr_data - X[j, :]
                    data_dist = np.linalg.norm(data_diff)
                    print data_dist
                    curr_distances.append(data_dist/scales[y[i]])
        distances.append(min(curr_distances))
    return distances




def main(target_model_name):
    np.random.seed(0)
    tf.set_random_seed(0)

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    dim = int(FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS*FLAGS.NUM_CHANNELS)

    _, _, X_test, Y_test = data_mnist()
    print('Loaded data')

    # target model for crafting adversarial examples
    target_model = load_model(target_model_name)
    target_model_name = basename(target_model_name)

    logits = target_model(x)
    prediction = K.softmax(logits)

    sess = tf.Session()
    print('Creating session')

    Y_test_uncat = np.argmax(Y_test,1)

    means, class_frac = class_means(X_test, Y_test_uncat)

    scales, mean_dists, closest_means = length_scales(X_test, Y_test_uncat)

    if args.norm == 'linf':
        eps_list = list(np.linspace(0.0, 0.1, 5))
        eps_list.extend(np.linspace(0.2, 0.5, 7))
        # eps_list = [0.3]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 9.0, 28))
        # eps_list = [6.0]

    # ofile = open('output_data/baseline_'+args.norm+'_md_rand_'+str(args.alpha)+'_'+str(target_model_name)+'.txt', 'a')

    for eps in eps_list:
        eps_orig = eps
        if args.alpha > eps:
            alpha = eps
            eps = 0
        elif eps >= args.alpha:
            alpha = args.alpha
            eps -= args.alpha

        adv_success = 0.0
        avg_l2_perturb = 0.0
        for i in range(FLAGS.NUM_CLASSES):
            curr_indices = np.where(Y_test_uncat == i)
            X_test_ini = X_test[curr_indices]
            Y_test_curr = Y_test_uncat[curr_indices]
            curr_len = len(X_test_ini)

            random_perturb = np.random.randn(*X_test_ini.shape)

            if args.norm == 'linf':
                random_perturb_signed = np.sign(random_perturb)
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_signed, CLIP_MIN, CLIP_MAX)
            elif args.norm == 'l2':
                random_perturb_unit = random_perturb/np.linalg.norm(random_perturb.reshape(curr_len,dim), axis=1)[:, None, None, None]
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

            closest_class = int(closest_means[i])

            mean_diff_vec = means[closest_class] - means[i]

            if args.norm == 'linf':
                mean_diff_vec_signed = np.sign(mean_diff_vec)
                perturb = eps  * mean_diff_vec_signed
            elif args.norm == 'l2':
                mean_diff_vec_unit = mean_diff_vec/np.linalg.norm(mean_diff_vec.reshape(dim))
                perturb = eps * mean_diff_vec_unit

            X_adv = np.clip(X_test_curr + perturb, CLIP_MIN, CLIP_MAX)

            # Getting the norm of the perturbation
            perturb_norm = np.linalg.norm((X_adv-X_test_ini).reshape(curr_len, dim), axis=1)
            perturb_norm_batch = np.mean(perturb_norm)
            avg_l2_perturb += perturb_norm_batch

            predictions_adv = K.get_session().run([prediction], feed_dict={x: X_adv, K.learning_phase(): 0})[0]

            adv_success += np.sum(np.argmax(predictions_adv, 1) != Y_test_curr)

            for k in range(2):
                adv_label = np.argmax(predictions_adv[k].reshape(1, FLAGS.NUM_CLASSES),1)
                img.imsave( 'images/baseline/'+args.norm+'/md_{}_{}_{}_{}_{}_{}.png'.format(
                        i, k, adv_label, closest_class, eps, alpha),
                        X_adv[k].reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)*255, cmap='gray')
        err = 100.0 * adv_success/ len(X_test)
        avg_l2_perturb = avg_l2_perturb/FLAGS.NUM_CLASSES

        print('{}, {}, {}'.format(eps, alpha, err))
        print('{}'.format(avg_l2_perturb))
        # ofile.write('{:.2f} {:.2f} {:.2f} {:.2f} \n'.format(eps, alpha, err, avg_l2_perturb))
    # ofile.write('\n \n')
    # ofile.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    # parser.add_argument("--eps", type=float, default=0.3,
    #                         help="FGS attack scale")
    parser.add_argument("--norm", type=str, default='linf',
                            help="Norm constraint to use")
    parser.add_argument("--alpha", type=float, default=0.0,
                            help="Amount of randomness")

    args = parser.parse_args()

    set_mnist_flags()

    main(args.target_model)
