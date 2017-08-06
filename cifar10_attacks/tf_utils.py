import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from attack_utils import gen_adv_loss
from keras.models import save_model

import time
import sys

FLAGS = flags.FLAGS
EVAL_FREQUENCY = 1000
BATCH_SIZE = 64
BATCH_EVAL_NUM = 100

def batch_eval(tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    From: https://github.com/openai/cleverhans/blob/master/cleverhans/utils_tf.py
    """

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m

    out = []
    for _ in tf_outputs:
        out.append([])

    for start in range(0, m, BATCH_SIZE):
        batch = start // BATCH_SIZE

        # Compute batch start and end indices
        start = batch * BATCH_SIZE
        end = start + BATCH_SIZE
        numpy_input_batches = [numpy_input[start:end]
                               for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= BATCH_SIZE
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        feed_dict = dict(zip(tf_inputs, numpy_input_batches))
        feed_dict[K.learning_phase()] = 0
        numpy_output_batches = K.get_session().run(tf_outputs,
                                                   feed_dict=feed_dict)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def tf_test_error_rate(logits, x, X_test, y_test):
    """
    Compute test error.
    """
    assert len(X_test) == len(y_test)

    # Predictions for the test set
    eval_prediction = K.softmax(logits)

    predictions = batch_eval([x], [eval_prediction], [X_test])[0]

    return error_rate(predictions, y_test)



def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)

    preds = np.argmax(predictions, 1)

    orig = np.argmax(labels, 1)

    error_rate = 100.0 - (100.0 * np.sum(preds == orig) / predictions.shape[0])

    return preds, orig, error_rate
