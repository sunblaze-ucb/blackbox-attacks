# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import random

import cifar_input_ensemble
import numpy as np
import resnet_model_reusable_wide
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_float('epsilon', 8.,
                          'Strength of adversarial perturbation. (0 to 255)')


def train(hps):
  """Training loop."""
  images, images_adv_thin, images_adv_wide, images_adv_tutorial, labels = cifar_input_ensemble.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  images_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images)

  def make_dynamic_adv_images():
    adv_model = resnet_model_reusable_wide.ResNet(hps, images_scaled, None, 'eval')
    adv_model._build_model()
    # Predict labels to use in no-label-leaking adversarial examples. Not in training mode.
    adv_model.labels = tf.stop_gradient(tf.one_hot(tf.argmax(adv_model.logits, axis=1), depth=hps.num_classes))
    # Generate adversarial examples. Not in training mode.
    adv_model._build_cost()
    grads, = tf.gradients(adv_model.cost, images, name='gradients_fgsm')
    perturbation = FLAGS.epsilon * tf.sign(grads)
    dynamic_adv_images = tf.stop_gradient(tf.clip_by_value(images + perturbation, 0., 255.))
    return dynamic_adv_images
  # The process freezes if we really do this conditionally.
  dynamic_adv_images = make_dynamic_adv_images()

  # Choose a set of adversarial examples at random.
  adv_choice = tf.random_uniform([], maxval=3, dtype=tf.int32, name='adv_choice')
  poison = tf.fill([hps.batch_size, 32, 32, 3], float('NaN'))
  adv_images = tf.case([
    (tf.equal(adv_choice, 0), lambda: dynamic_adv_images),
    (tf.equal(adv_choice, 1), lambda: images_adv_thin),
    (tf.equal(adv_choice, 2), lambda: images_adv_wide),
    # images_adv_tutorial is held out
  ], default=lambda: poison)
  tf.summary.image('adv_images', adv_images)
  adv_images_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), adv_images)
  # Train on benign and adversarial images.
  # This is equivalent to averaging the costs of two copies of the model.
  grand_images_scaled = tf.concat([images_scaled, adv_images_scaled], axis=0)
  grand_labels = tf.concat([labels, labels], axis=0)
  grand_hps = hps._replace(batch_size=hps.batch_size*2)
  model = resnet_model_reusable_wide.ResNet(grand_hps, grand_images_scaled, grand_labels, FLAGS.mode, reuse_variables=True)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  eq_float = tf.to_float(tf.equal(predictions, truth))
  precision_benign = tf.reduce_mean(eq_float[:hps.batch_size])
  precision_adv = tf.reduce_mean(eq_float[hps.batch_size:])
  precision = (precision_benign + precision_adv) / 2

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision),
                                   tf.summary.scalar('Precision_benign', precision_benign),
                                   tf.summary.scalar('Precision_adversarial', precision_adv)]))

  stop_hook = tf.train.StopAtStepHook(last_step=80000)

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision,
               'precision_benign': precision_benign,
               'precision_adv': precision_adv},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook, stop_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def evaluate(hps):
  """Eval loop."""
  images, labels = cifar_input_nostd.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  images = tf.map_fn(lambda image: tf.image.per_image_standardization(image), images)
  model = resnet_model_reusable_wide.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model_reusable_wide.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
