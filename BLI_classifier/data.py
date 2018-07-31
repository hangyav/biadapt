from __future__ import division
from __future__ import print_function

import codecs
import math
import random


class BilexDataFeeder(object):
  def __init__(self, data_file, batch_size, shuffle=False, filter=None):
    self._data_file = data_file
    self.epoch = 0
    self._batch_size = batch_size
    self._examples_scanner = self.scan_examples()
    self.num_examples_fed = 0
    with codecs.open(data_file, 'rb', 'utf-8') as f:
      self.num_examples_epoch = len([0 for _ in f])
    self._shuffle = shuffle
    self._rg = random.Random(123)
    self._filter = filter
    self._examples = self.read_examples()

  @property
  def name(self):
    return self._data_file

  def read_examples(self):
    num_examples = 0
    num_examples_used = 0
    with codecs.open(self._data_file, 'rb', 'utf-8') as f:
      examples = []
      for l in f:
        w_s, w_t = tuple(l.strip().split('\t'))
        num_examples += 1
        if self._filter and not self._filter(w_s, w_t):
          continue
        examples.append((w_s, w_t))
        num_examples_used += 1
    print('Using', num_examples_used, 'of', num_examples, 'examples.')
    return examples

  def get_lexicon(self):
    return list(self._examples)

  def scan_examples(self):
    examples = self._examples
    if self._shuffle:
      self._rg.shuffle(examples)
    for example in examples:
      yield example

  def get_batch(self):
    examples = []
    i = 0
    while i < self._batch_size:
      try:
        examples.append(next(self._examples_scanner))
        self.num_examples_fed += 1
        i += 1
      except StopIteration:
        self.epoch += 1
        self._examples_scanner = self.scan_examples()
    return examples


class SkipgramDataFeeder(object):
  def __init__(self, data_file, batch_size, wordcounts, window_size, subsample, shuffle=True):
    self._data_file = data_file
    self.epoch = 0
    self._batch_size = batch_size
    self._examples_scanner = self.scan_examples()
    self.num_examples_fed = 0
    with codecs.open(data_file, 'rb', 'utf-8') as f:
      self.num_examples_epoch = len([0 for _ in f])
    self._shuffle = shuffle
    self._window_size = window_size
    self._rg = random.Random(123)
    self._subsampler = Word2VecSubsampler(wordcounts, subsample, self._rg)
    self._sentences = self.read_sentences()

  def read_sentences(self):
    with codecs.open(self._data_file, 'rb', 'utf-8') as f:
      sentences = []
      for l in f:
        s = l.strip().split()
        sentences.append(s)
    return sentences

  def scan_examples(self):
    sentences = self._sentences
    if self._shuffle:
      self._rg.shuffle(sentences)
    for s in sentences:
      _, center_words, context_words_s = \
        skipgram_preprocessing(s, self._window_size, self._rg, self._subsampler)
      for center_word, context_words in zip(center_words, context_words_s):
        for context_word in context_words:
          yield center_word, context_word

  def get_batch(self):
    examples = []
    i = 0
    while i < self._batch_size:
      try:
        examples.append(next(self._examples_scanner))
        self.num_examples_fed += 1
        i += 1
      except StopIteration:
        self.epoch += 1
        self._examples_scanner = self.scan_examples()
    return examples


def skipgram_preprocessing(sentence, window_size, rg, subsampler):
  words = [w for w in sentence if not subsampler.subsample(w)]
  center_words = []
  context_words_sen = []
  for center_pos, center_word in enumerate(words):
    effective_window = rg.randint(1, window_size)
    start = max(0, center_pos - effective_window)
    end = min(len(words), center_pos + effective_window + 1)
    context_words = []
    for context_pos in xrange(start, end):
      if context_pos == center_pos:
        continue
      context_words.append(words[context_pos])
    if context_words:
      center_words.append(center_word)
      context_words_sen.append(context_words)
  return len(words), center_words, context_words_sen


class Word2VecSubsampler(object):
  def __init__(self, wordcounts, subsample, rg):
    self._wordcounts = wordcounts
    self._corpus_size = sum(wordcounts.values())
    self._subsample = subsample
    self._rg = rg
    self._num_subsampled = 0
    self._num_processed = 0

  def subsample(self, word):
    if word not in self._wordcounts:
      return True
    freq = self._wordcounts[word]
    keep_prob = (math.sqrt(freq / (self._subsample * self._corpus_size)) + 1) * \
                ((self._subsample * self._corpus_size) / freq)
    subsample = self._rg.random() > keep_prob
    if subsample:
      self._num_subsampled += 1
    self._num_processed += 1
    return subsample

  def subsample_percentage(self):
    return self._num_subsampled / self._num_processed
