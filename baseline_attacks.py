import numpy as np
import tensorflow as tf
import keras.backend as K
from mnist import data_mnist, set_mnist_flags, load_model
from os.path import basename

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

CLIP_MIN = 0
CLIP_MAX = 1

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

    dim = int(FLAGS.IMAGE_ROWS*FLAGS.IMAGE_COLS)

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

    adv_success = 0.0
    for i in range(FLAGS.NUM_CLASSES):
        curr_indices = np.where(Y_test_uncat == i)
        X_test_curr = X_test[curr_indices]
        Y_test_curr = Y_test_uncat[curr_indices]

        mean_diff_vec = means[int(closest_means[i])] - means[i]

        mean_diff_vec_signed = np.sign(mean_diff_vec)

        X_adv = np.clip(X_test_curr + args.eps * mean_diff_vec_signed, CLIP_MIN, CLIP_MAX)

        predictions_adv = K.get_session().run([prediction], feed_dict={x: X_adv, K.learning_phase(): 0})[0]

        adv_success += np.sum(np.argmax(predictions_adv, 1) != Y_test_curr)

    print('{}'.format(adv_success/ len(X_test)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--eps", type=float, default=0.3,
                            help="FGS attack scale")

    args = parser.parse_args()

    set_mnist_flags()

    main(args.target_model)
