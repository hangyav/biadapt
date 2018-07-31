from __future__ import division
from __future__ import print_function

import codecs
import sys
import time
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from evaluation import eval_translations, Result, extract_translations


class Trainer(object):
  def __init__(self, model, num_epochs, data_feeder):
    self._model = model
    self._num_epochs = num_epochs
    self._data_feeder = data_feeder
    self._tasks = []

  def add_command(self, command):
    self._tasks.append(command)

  def train(self):
    start_time = time.time()
    for task in self._tasks:
      task.set_start_time(start_time)
    current_epoch = 0
    while current_epoch < self._num_epochs:
      data_feed = self._data_feeder.get_batch()
      self._model.step(data_feed)
      end_time = time.time()
      current_epoch = self._data_feeder.epoch
      for task in self._tasks:
        task.maybe_execute(current_epoch, end_time)


class SemiSupTrainer(object):
  def __init__(self, model, num_epochs, data_feeder, unlabeled_data_feeder):
    self._model = model
    self._num_epochs = num_epochs
    self._data_feeder = data_feeder
    self._unlabeled_data_feeder = unlabeled_data_feeder
    self._tasks = []

  def add_command(self, command):
    self._tasks.append(command)

  def train(self):
    start_time = time.time()
    for task in self._tasks:
      task.set_start_time(start_time)
    current_epoch = 0
    while current_epoch < self._num_epochs:
      data_feed = self._data_feeder.get_batch()
      unlabeled_data_feed = self._unlabeled_data_feeder.get_batch()
      self._model.step({'labeled': data_feed, 'unlabeled': unlabeled_data_feed})
      end_time = time.time()
      current_epoch = self._data_feeder.epoch
      for task in self._tasks:
        task.maybe_execute(current_epoch, end_time)


class MultitaskTrainer(object):
  def __init__(self, model, num_epochs, data_feeders):
    self._model = model
    self._num_epochs = num_epochs
    self._data_feeders = data_feeders
    self._tasks = []

  def add_command(self, command):
    self._tasks.append(command)

  def train(self):
    start_time = time.time()
    for task in self._tasks:
      task.set_start_time(start_time)
    current_epochs = 0
    while current_epochs < self._num_epochs:
      data_feeds = [f.get_batch() for f in self._data_feeders]
      self._model.step(data_feeds)
      end_time = time.time()
      current_epochs = self._data_feeders[0].epoch
      for task in self._tasks:
        task.maybe_execute(current_epochs, end_time)


class SemiSupEpochLossLogger(object):
  def __init__(self, model, log_dir):
    self._model = model
    with tf.variable_scope('losses'):
      self._epoch_loss = tf.placeholder(dtype='float32', shape=[], name='epoch_loss')
      self._walker_loss = tf.placeholder(dtype='float32', shape=[], name='walker_loss')
      self._visit_loss = tf.placeholder(dtype='float32', shape=[], name='visit_loss')
      self._logit_loss = tf.placeholder(dtype='float32', shape=[], name='logit_loss')
      self._epoch_summary_op = tf.summary.scalar('epoch_loss', self._epoch_loss)
      self._walker_summary_op = tf.summary.scalar('walker_loss', self._walker_loss)
      self._visit_summary_op = tf.summary.scalar('visit_loss', self._visit_loss)
      self._logit_summary_op = tf.summary.scalar('logit_loss', self._logit_loss)
      self._summary_op = tf.summary.merge([self._epoch_summary_op, self._walker_summary_op, self._visit_summary_op,
                                           self._logit_summary_op])
    self._epoch = 0
    self._losses_epoch = []
    self._losses_walker = []
    self._losses_visit = []
    self._losses_logit = []
    self._summary_writer = tf.summary.FileWriter(log_dir)
    self._update_time = 0

  def set_start_time(self, time):
    self._update_time = time

  def maybe_execute(self, epoch, time):
    loss = self._model.loss()
    self._losses_epoch.append(loss[0])
    self._losses_walker.append(loss[1])
    self._losses_visit.append(loss[2])
    self._losses_logit.append(loss[3])
    if epoch > self._epoch:
      average_loss = sum(self._losses_epoch) / len(self._losses_epoch)
      average_loss_walker = sum(self._losses_walker) / len(self._losses_walker)
      average_loss_visit = sum(self._losses_visit) / len(self._losses_visit)
      average_loss_logit = sum(self._losses_logit) / len(self._losses_logit)
      summ_str = self._model.session.run(self._summary_op,
                                         {
                                           self._epoch_loss: average_loss,
                                           self._walker_loss: average_loss_walker,
                                           self._visit_loss: average_loss_visit,
                                           self._logit_loss: average_loss_logit
                                         })
      self._summary_writer.add_summary(summ_str, epoch)
      self._losses_epoch = []
      self._losses_walker = []
      self._losses_visit = []
      self._losses_logit = []
      self._epoch = epoch


class EpochLossLogger(object):
  def __init__(self, model, log_dir):
    self._model = model
    self._epoch_loss = tf.placeholder(dtype='float32', shape=[], name='epoch_loss')
    self._summary_op = tf.summary.scalar('epoch_loss', self._epoch_loss)
    self._epoch = 0
    self._losses_epoch = []
    self._summary_writer = tf.summary.FileWriter(log_dir)
    self._update_time = 0

  def set_start_time(self, time):
    self._update_time = time

  def maybe_execute(self, epoch, time):
    self._losses_epoch.append(self._model.loss())
    if epoch > self._epoch:
      average_loss = sum(self._losses_epoch) / len(self._losses_epoch)
      summ_str = self._model.session.run(self._summary_op, {self._epoch_loss: average_loss})
      self._summary_writer.add_summary(summ_str, epoch)
      self._losses_epoch = []
      self._epoch = epoch


class MultitaskEpochLossLogger(object):
  def __init__(self, model, log_dir):
    self._model = model
    num_tasks = model.num_tasks()
    self._epoch_loss = [tf.placeholder(dtype='float32', shape=[], name='epoch_loss') for _ in xrange(num_tasks)]
    self._summary_ops = [tf.summary.scalar(('epoch_loss_%i' % i), self._epoch_loss[i]) for i in xrange(num_tasks)]
    self._epoch = 0
    self._losses_epoch = []
    self._summary_writer = tf.summary.FileWriter(log_dir)
    self._update_time = 0

  def set_start_time(self, time):
    self._update_time = time

  def maybe_execute(self, epoch, time):
    self._losses_epoch.append(self._model.losses())
    if epoch > self._epoch:
      for i, summary_op in enumerate(self._summary_ops):
        losses = [t[i] for t in self._losses_epoch]
        average_loss = sum(losses) / len(losses)
        summ_str = self._model.session.run(summary_op, {self._epoch_loss[i]: average_loss})
        self._summary_writer.add_summary(summ_str, epoch)
      self._losses_epoch = []
      self._epoch = epoch


class BasicStatsLogger(object):
  def __init__(self, model, data_feeder, num_epochs, period):
    self._model = model
    self._data_feeder = data_feeder
    self._num_epochs = num_epochs
    self._period = period
    self._update_time = 0
    self._losses = []

  def set_start_time(self, time):
    self._update_time = time

  def maybe_execute(self, epoch, time):
    loss = self._model.loss()
    if np.isnan(loss):
      print('Loss is nan. Last two batches:')
      print(self._model.last_examples())
      raise NanLoss
    self._losses.append(loss)
    if time - self._update_time > self._period:
      num_steps = len(self._losses)
      average_loss = sum(self._losses) / num_steps
      step_frequency = num_steps / (time - self._update_time)
      progress = self._data_feeder.num_examples_fed / \
                 (self._data_feeder.num_examples_epoch * self._num_epochs) * 100
      print('\rprogress %3.2f %%, loss %6.4f, step rate %3.3f steps/s' %
            (progress, average_loss, step_frequency), end='')
      sys.stdout.flush()
      self._update_time = time
      self._losses = []


class SemiSupBasicStatsLogger(object):
  def __init__(self, model, data_feeder, num_epochs, period):
    self._model = model
    self._data_feeder = data_feeder
    self._num_epochs = num_epochs
    self._period = period
    self._update_time = 0
    self._losses = []

  def set_start_time(self, time):
    self._update_time = time

  def maybe_execute(self, epoch, time):
    loss = self._model.loss()[0]
    if np.isnan(loss):
      print('Loss is nan. Last two batches:')
      print(self._model.last_examples())
      raise NanLoss
    self._losses.append(loss)
    if time - self._update_time > self._period:
      num_steps = len(self._losses)
      average_loss = sum(self._losses) / num_steps
      step_frequency = num_steps / (time - self._update_time)
      progress = self._data_feeder.num_examples_fed / \
                 (self._data_feeder.num_examples_epoch * self._num_epochs) * 100
      print('\rprogress %3.2f %%, loss %6.4f, step rate %3.3f steps/s' %
            (progress, average_loss, step_frequency), end='')
      sys.stdout.flush()
      self._update_time = time
      self._losses = []


class NanLoss(Exception):
  pass


class Evaluation(object):
  def __init__(self, model, data_feeder, training_lexicon, num_epochs, log_dir, modes=('all',)):
    self._model = model
    self._data_feeder = data_feeder
    self._training_lexicon = {}
    for w_s, w_t in training_lexicon:
      if w_s in training_lexicon:
        self._training_lexicon[w_s].add(w_t)
      else:
        self._training_lexicon[w_s] = {w_t}
    self._epoch = 0
    self._num_epochs = num_epochs
    self._log_dir = log_dir
    with tf.variable_scope('evaluation') as scope:
      self._result_top_placeholders = self._init_summaries('top', modes)
      self._result_all_placeholders = self._init_summaries('all', modes)
      graph = tf.get_default_graph()
      summaries_all = graph.get_collection(tf.GraphKeys.SUMMARIES)
      print(summaries_all)
      summaries = graph.get_collection(tf.GraphKeys.SUMMARIES, scope='evaluation')
      print(summaries)
      #summaries = if s.scope ]
      self._summary_op = tf.summary.merge(summaries)
    self._summary_writer = tf.summary.FileWriter(log_dir)
    self._modes = modes

  def _init_summaries(self, model_mode, modes):
    result_placeholders = {}
    dtype = 'float32'
    shape = []
    for eval_mode in modes:
      suffix = '_%s_%s' % (model_mode, eval_mode)
      f1_top = tf.placeholder(dtype, shape, 'f1' + suffix)
      precision_top = tf.placeholder(dtype, shape, 'precision' + suffix)
      recall_top = tf.placeholder(dtype, shape, 'recall' + suffix)
      result_placeholder = Result(f1_top, precision_top, recall_top)
      tf.summary.scalar('f1' + suffix, f1_top)
      tf.summary.scalar('precison' + suffix, f1_top)
      tf.summary.scalar('recall' + suffix, f1_top)
      result_placeholders[eval_mode] = result_placeholder
    return result_placeholders

  def set_start_time(self, time):
    return

  def maybe_execute(self, epoch, time):
    if epoch > self._epoch + 3:
      threshold = 0.5
      summary_feeds = {}
      for mode in self._modes:  # mode can be all, uniword, multiword
        source_words, target_words, predicted_targets = self.predict_translations(threshold, mode)
        oov_words = self._model.get_oov_words(target_words)
        result_top, result_all = self.evaluate(source_words, target_words, predicted_targets)
        self._print_eval(result_top, result_all, oov_words, predicted_targets, target_words, mode)
        summary_feeds.update(
          {self._result_top_placeholders[mode].precision: result_top.precision,
           self._result_top_placeholders[mode].recall: result_top.recall,
           self._result_top_placeholders[mode].f1: result_top.f1,
           self._result_all_placeholders[mode].precision: result_all.precision,
           self._result_all_placeholders[mode].recall: result_all.recall,
           self._result_all_placeholders[mode].f1: result_all.f1})
      summary = self._model.session.run(self._summary_op, summary_feeds)
      self._summary_writer.add_summary(summary, epoch)
      self._epoch = epoch

    if epoch == self._num_epochs:
      thresholds = [t/10 for t in xrange(0, 10, 1)] + [0.92, .94, .96, .98]
      for mode in self._modes:
        results_top, results_all, predicted_translations = self._eval_thresholds(thresholds, mode)
        self._predicted_translations_to_file(predicted_translations, thresholds, mode)
        self._results_to_file(results_top, results_all, mode)

  def _results_to_file(self, results_top, results_all, mode):
    df = pd.DataFrame(
      {'precision_1': results_top.precision,
       'precision_all': results_all.precision,
       'recall_1': results_top.recall,
       'recall_all': results_all.recall,
       'f1_1': results_top.f1,
       'f1_all': results_all.f1}
    )
    results_file = path.join(self._log_dir, 'results.%s.csv' % mode)
    df.to_csv(results_file)

  def _predicted_translations_to_file(self, predicted_translations, thresholds, mode):
    predicted_translations_f = path.join(self._log_dir, 'translations.%s.txt' % mode)
    with codecs.open(predicted_translations_f, 'wb', 'utf-8') as f:
      for t, predicted_translations_t in zip(thresholds, predicted_translations):
        f.write('threshold %1.1f\n' % t)
        for ts in predicted_translations_t:
          ts = [t[0] for t in ts]
          f.write('\t'.join(ts))
          f.write('\n')
        f.write('\n')

  def _eval_thresholds(self, thresholds, mode):
    prec_1_vals, prec_all_vals = np.zeros(shape=(len(thresholds))), np.zeros(shape=(len(thresholds)))
    rec_1_vals, rec_all_vals = np.zeros(shape=(len(thresholds))), np.zeros(shape=(len(thresholds)))
    f1_1_vals, f1_all_vals = np.zeros(shape=(len(thresholds))), np.zeros(shape=(len(thresholds)))
    predicted_targets = []
    for i, threshold in enumerate(thresholds):
      source_words, target_words, predicted_targets_threshold = self.predict_translations(threshold, mode)
      result_top, result_all = self.evaluate(source_words, target_words, predicted_targets_threshold)
      prec_1_vals[i] = result_top.precision
      prec_all_vals[i] = result_all.precision
      rec_1_vals[i] = result_top.recall
      rec_all_vals[i] = result_all.recall
      f1_1_vals[i] = result_top.f1
      f1_all_vals[i] = result_all.f1
      predicted_targets.append(predicted_targets_threshold)
    results_top = Result(f1_1_vals, prec_1_vals, rec_1_vals)
    results_all = Result(f1_all_vals, prec_all_vals, rec_all_vals)
    return results_top, results_all, predicted_targets

  def _print_eval(self, result_top, result_all,  oov_words, predicted_translations, target_words, mode):
    print()
    print('Evaluation of %s on %s' % (mode, self._data_feeder.name))
    print(round(len(oov_words) / len(target_words) * 100, 2), '% out of training vocabulary.')
    print('predicted @1: ')
    for w_pred, w_true in zip(predicted_translations, target_words):
      string1 = w_true + ':'
      string2 = w_pred[0][0] + ',' + str(w_pred[0][1]) if w_pred else ''
      string = string1 + string2
      print(string.encode('utf-8'), ' ', end='')
    print()
    print('recall@1: %3.2f' % result_top.recall)
    print('precision@1: %3.2f' % result_top.precision)
    print('f1@1: %3.2f' % result_top.f1)
    print('recall@all: %3.2f' % result_all.recall)
    print('precision@all: %3.2f' % result_all.precision)
    print('f1@all: %3.2f' % result_all.f1)

  def evaluate(self, source_words, target_words, predicted_targets):
    top_1_translation_pairs, translation_pairs = extract_translations(
      self._training_lexicon, source_words, predicted_targets)
    groundtruth = zip(source_words, target_words)
    result_top = eval_translations(groundtruth, top_1_translation_pairs)
    result_all = eval_translations(groundtruth, translation_pairs)
    return result_top, result_all

  def predict_translations(self, threshold, mode):
    data_feeder = self._data_feeder
    eval_data_epoch = data_feeder.epoch
    predict_targets = []
    source_words = []
    target_words = []
    while data_feeder.epoch == eval_data_epoch:
      data_feed = data_feeder.get_batch()
      source_words_batch = []
      target_words_batch = []
      for word_s, word_t in data_feed:
        is_mwu_pair = ' ' in word_s or ' ' in word_t
        if mode == 'mwu':
          if is_mwu_pair:
            source_words_batch.append(word_s)
            target_words_batch.append(word_t)
        elif mode == 'uniword':
          if not is_mwu_pair:
            source_words_batch.append(word_s)
            target_words_batch.append(word_t)
        else:
          source_words_batch.append(word_s)
          target_words_batch.append(word_t)

      source_words.extend(source_words_batch)
      target_words.extend(target_words_batch)
      predicted_targets_batch = self._model.predict(
        source_words_batch, source2target=True, threshold=threshold)
      predict_targets.extend(predicted_targets_batch)

    return source_words, target_words, predict_targets
