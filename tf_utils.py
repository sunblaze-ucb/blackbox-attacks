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
BATCH_SIZE = 100
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


def tf_train(x, y, model, X_train, Y_train, generator, x_advs=None, benign = None, cross_lip=None):
    old_vars = set(tf.all_variables())
    train_size = Y_train.shape[0]

    # Generate cross-entropy loss for training
    logits = model(x)
    #print(K.int_shape(logits))
    preds = K.softmax(logits)
    l1 = gen_adv_loss(logits, y, mean=True)

    # add adversarial training loss
    if x_advs is not None:
        idx = tf.placeholder(dtype=np.int32)
        logits_adv = model(tf.stack(x_advs)[idx])
        l2 = gen_adv_loss(logits_adv, y, mean=True)
        if benign == 0:
            loss = l2
        elif benign == 1:
            loss = 0.5*(l1+l2)
    elif cross_lip is not None:
        logit_grad = K.gradients(logits, [x])[1]
	print K.int_shape(logit_grad)
	l3 = tf.zeros([])
	for i in range(10):
	    logit_slice = tf.slice(logits, [0,i], [1,1])
	   # print(K.int_shape(logit_slice))
	    logit_loss = tf.reshape(logit_slice,[1])
	    #print(K.int_shape(logit_loss))
	    logit_slice_grad = K.gradients(logit_loss, [x])[0]
	    #print(K.int_shape(logit_slice_grad))
	    l3 = tf.add(l3, tf.reduce_sum(tf.square(logit_slice_grad)))
	l3 = K.mean(l3)
	logit_grad_norm = tf.reduce_sum(tf.square(logit_grad))
	l2 = K.mean(logit_grad_norm)
	loss = l1 + 10e-5 * l2
    else:
        l2 = tf.constant(0)
        loss = l1

    #return

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # saver = tf.train.Saver(set(tf.all_variables()) - old_vars)

    # Run all the initializers to prepare the trainable parameters.
    K.get_session().run(tf.initialize_variables(
        set(tf.all_variables()) - old_vars))
    start_time = time.time()
    print('Initialized!')

    # Loop through training steps.
    num_steps = int(FLAGS.NUM_EPOCHS * train_size + BATCH_SIZE - 1) // BATCH_SIZE

    step = 0
    training_loss = 0
    epoch_count = 0
    step_old = 0
    for (batch_data, batch_labels) \
            in generator.flow(X_train, Y_train, batch_size=BATCH_SIZE):

        if len(batch_data) < BATCH_SIZE:
            k = BATCH_SIZE - len(batch_data)
            batch_data = np.concatenate([batch_data, X_train[0:k]])
            batch_labels = np.concatenate([batch_labels, Y_train[0:k]])

        feed_dict = {x: batch_data,
                     y: batch_labels,
                     K.learning_phase(): 1}

        # choose source of adversarial examples at random
        # (for ensemble adversarial training)
        if x_advs is not None:
            feed_dict[idx] = np.random.randint(len(x_advs))

        # Run the graph
        _, curr_loss, curr_l1, curr_l2, curr_preds, _ = \
            K.get_session().run([optimizer, loss, l1, l2, preds]
                                + [model.updates],
                                feed_dict=feed_dict)
        training_loss += curr_loss

        epoch = float(step) * BATCH_SIZE / train_size
        if epoch >= epoch_count:
            epoch_count += 1
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.2f s' %
                (step, float(step) * BATCH_SIZE / train_size,
                 elapsed_time))
            print('Training loss: %.3f' % (training_loss/(step - step_old)))
            training_loss = 0
            step_old = step
            print('Minibatch loss: %.3f (%.3f, %.3f)' % (curr_loss, curr_l1, curr_l2))

            print('Minibatch error: %.1f%%' % error_rate(curr_preds, batch_labels))

        # if step % EVAL_FREQUENCY == 0:
        #     elapsed_time = time.time() - start_time
        #     start_time = time.time()
        #     print('Step %d (epoch %.2f), %.2f s' %
        #         (step, float(step) * BATCH_SIZE / train_size,
        #          elapsed_time))
            # print('Minibatch loss: %.3f (%.3f, %.3f)' % (curr_loss, curr_l1, curr_l2))
            #
            # print('Minibatch error: %.1f%%' % error_rate(curr_preds, batch_labels))

            # # save_path = saver.save(K.get_session(), "~/blackbox-attacks/tmp/model.ckpt")
            # save_model(model, 'tmp/model.ckpt')
            # print("Model saved in file: %s" % 'model.ckpt')
            # for (val_data, val_labels) in generator.flow(X_train, Y_train, batch_size=50000):
            #             if len(batch_data) < BATCH_SIZE:
            #                 k = BATCH_SIZE - len(batch_data)
            #                 batch_data = np.concatenate([batch_data, X_train[0:k]])
            #                 batch_labels = np.concatenate([batch_labels, Y_train[0:k]])
            #     val_dict = {x: val_data,
            #                 y: val_labels,
            #                 K.learning_phase(): 0}
            # val_loss, val_preds = K.get_session().run([loss, preds], feed_dict=val_dict)


            # print('Training error: %.1f%%' % error_rate(val_preds, Y_train))
                # break


            sys.stdout.flush()

        step += 1
        if step == num_steps:
            break


def tf_test_error_rate(model, x, X_test, y_test):
    """
    Compute test error.
    """
    assert len(X_test) == len(y_test)

    # Predictions for the test set
    eval_prediction = K.softmax(model(x))

    predictions = batch_eval([x], [eval_prediction], [X_test])[0]

    return error_rate(predictions, y_test)


def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)

    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
