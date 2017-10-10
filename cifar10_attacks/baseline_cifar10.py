import numpy as np
import tensorflow as tf
import keras.backend as K
import models
from os.path import basename
from matplotlib import image as img
# from cifar10_setup import set_cifar10_flags


NUM_CLASSES = 10
IMAGE_ROWS = 32
IMAGE_COLS = 32
NUM_CHANNELS = 3
BATCH_SIZE_G = 1000

CLIP_MIN = 0
CLIP_MAX = 255
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

    x = tf.placeholder(shape=(None,
                       IMAGE_ROWS,
                       IMAGE_COLS,
                       NUM_CHANNELS),dtype=tf.float32)
    y = tf.placeholder(tf.int64, shape=None)

    dim = int(IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS)

    X_test = np.load(args.img_source)
    X_test_batches = X_test.reshape((-1, BATCH_SIZE_G, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    Y_test = np.load(args.label_source)
    Y_test_batches = Y_test.reshape((-1, BATCH_SIZE_G, NUM_CLASSES))
    print('Loaded data')


    # i = 6711
    # img.imsave( 'images/cifar10_{}.png'.format(i),
    #             X_test[i]/255)

    Y_test_uncat = np.argmax(Y_test,1)
    Y_test_uncat_batches = Y_test_uncat.reshape((-1, BATCH_SIZE_G))

    # target model for crafting adversarial examples
    target_model = models.load_model('logs/'+target_model_name, BATCH_SIZE_G, x, y)

    logits = target_model.get_logits()
    prediction = tf.nn.softmax(logits)

    sess = tf.Session()
    target_model.load(sess)
    print('Creating session')

    # accuracy = target_model.get_accuracy()
    # accuracy_np = sess.run(accuracy,feed_dict={x: X_test, y: Y_test_uncat})
    # print(accuracy_np)

    # return
    
    benign_success = 0.0


    for i in range(len(X_test_batches)):
        predictions_np = sess.run(prediction, feed_dict={x: X_test_batches[i]})
        benign_success += np.sum(np.argmax(predictions_np, 1) != Y_test_uncat_batches[i])
    err = 100.0 * benign_success/10000.0
    print(err)

    means, class_frac = class_means(X_test, Y_test_uncat)

    scales, mean_dists, closest_means = length_scales(X_test, Y_test_uncat)

    if args.norm == 'linf':
        eps_list = list(np.linspace(0.0, 32.0, 9))
        # eps_list = [4]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 9.0, 28))
        # eps_list = [6.0]
    print eps_list

    if args.targeted_flag == 0:
        ofile = open('output_data/baseline_'+args.norm+'_md_rand_'+str(args.alpha)+'_'+str(target_model_name)+'.txt', 'a')
    elif args.targeted_flag == 1:
        ofile = open('output_data/baseline_target_'+args.norm+'_md_rand_'+str(args.alpha)+'_'+str(target_model_name)+'.txt', 'a')

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
        for i in range(NUM_CLASSES):
            curr_indices = np.where(Y_test_uncat == i)
            X_test_ini = X_test[curr_indices]
            Y_test_curr = Y_test_uncat[curr_indices]
            curr_len = len(X_test_ini)
            if args.targeted_flag == 1:
                allowed_targets = list(range(NUM_CLASSES))
                allowed_targets.remove(i)

            random_perturb = np.random.randn(*X_test_ini.shape)

            if args.norm == 'linf':
                random_perturb_signed = np.sign(random_perturb)
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_signed, CLIP_MIN, CLIP_MAX)
            elif args.norm == 'l2':
                random_perturb_unit = random_perturb/np.linalg.norm(random_perturb.reshape(curr_len,dim), axis=1)[:, None, None, None]
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

            if args.targeted_flag == 0:
                closest_class = int(closest_means[i])
                mean_diff_vec = means[closest_class] - means[i]
            elif args.targeted_flag == 1:
                targets = []
                mean_diff_array = np.zeros((curr_len, IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
                for j in range(curr_len):
                    target = np.random.choice(allowed_targets)
                    targets.append(target)
                    mean_diff_array[j] = means[target] - means[i]

            # img.imsave( 'images/cifar10_mean_{}.png'.format(i),
                        # means[i]/255)

            if args.norm == 'linf':
                if args.targeted_flag == 0:
                    mean_diff_vec_signed = np.sign(mean_diff_vec)
                    perturb = eps  * mean_diff_vec_signed
                elif args.targeted_flag == 1:
                    mean_diff_array_signed = np.sign(mean_diff_array)
                    perturb = eps  * mean_diff_array_signed
            elif args.norm == 'l2':
                mean_diff_vec_unit = mean_diff_vec/np.linalg.norm(mean_diff_vec.reshape(dim))
                perturb = eps * mean_diff_vec_unit

            X_adv = np.clip(X_test_curr + perturb, CLIP_MIN, CLIP_MAX)

            # Getting the norm of the perturbation
            perturb_norm = np.linalg.norm((X_adv-X_test_ini).reshape(curr_len, dim), axis=1)
            perturb_norm_batch = np.mean(perturb_norm)
            avg_l2_perturb += perturb_norm_batch

            predictions_adv = sess.run(prediction, feed_dict={x: X_adv})

            if args.targeted_flag == 0:
                adv_success += np.sum(np.argmax(predictions_adv, 1) != Y_test_curr)
            elif args.targeted_flag == 1:
                print(targets)
                adv_success += np.sum(np.argmax(predictions_adv, 1) == np.array(targets))

            # for k in range(1):
            #     adv_label = np.argmax(predictions_adv[k].reshape(1, NUM_CLASSES),1)
            #     img.imsave( 'images/baseline/'+args.norm+'/md_{}_{}_{}_{}_{}_{}.png'.format(
            #             i, k, adv_label, closest_class, eps, alpha),
            #             X_adv[k]/255)
        err = 100.0 * adv_success/ len(X_test)
        avg_l2_perturb = avg_l2_perturb/NUM_CLASSES

        print('{}, {}, {}'.format(eps, alpha, err))
        print('{}'.format(avg_l2_perturb))
        ofile.write('{:.2f} {:.2f} {:.2f} {:.2f} \n'.format(eps, alpha, err, avg_l2_perturb))
    # ofile.write('\n \n')
    ofile.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="target model for attack")
    parser.add_argument("--img_source", help="source of images",
                        default='test_orig.npy')
    parser.add_argument("--label_source", help="source of labels",
                        default='test_labels.npy')
    parser.add_argument("--norm", type=str, default='linf',
                            help="Norm constraint to use")
    parser.add_argument("--alpha", type=float, default=0.0,
                            help="Amount of randomness")
    parser.add_argument("--targeted_flag", type=int, default=0,
                            help="Carry out targeted attack")

    args = parser.parse_args()

    main(args.ckpt_dir)
