"""ResNet Train module.
"""
import time
import six
import sys
import os

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('mode', 'train', 'Train or evaluate')
tf.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.flags.DEFINE_string('train_data_path', 'cifar10/data_batch*', 'Filepattern for training data.')
tf.flags.DEFINE_integer('image_size', 32, 'Image side length.')


def train(hps, batch_size):
  """Training loop."""
  script_dir = os.path.dirname(__file__)

  images, orig_images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)

  if args.eps is not None:
    eps_num = args.eps
    eps = eps_num/255.
    print('{}'.format(eps))
  else:
    eps = args.eps

  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode, eps)
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
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'adv_loss': model.adv_cost,
               'precision': precision},
      every_n_iter=100)

  stop_hook = tf.train.StopAtStepHook(last_step=100000)

  saver = tf.train.Saver(max_to_keep=10)
  checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=log_root, saver=saver, save_steps = 10000)
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
      checkpoint_dir=log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[checkpoint_hook, summary_hook, stop_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_checkpoint_secs=None,
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)

def main():
    batch_size = 128

    # Assuming dataset is CIFAR-10
    num_classes = 10

    if 'wide' not in args.target_model:
        residual_count = 5
        wide_flag = False
    elif 'wide' in args.target_model:
        residual_count = 4
        wide_flag = True

    hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=residual_count,
                             wide_flag=wide_flag,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             adv_only=False,
                             optimizer='mom')

    train(hps, batch_size)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target_model', type=str,
                        help='name of target model')
    parser.add_argument("--eps", type=int, default=None,
                        help="FGS attack scale")

    args = parser.parse_args()
    log_root = 'logs_'+args.target_model
    train_dir = log_root+'/train'

    main()
