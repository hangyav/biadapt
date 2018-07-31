from __future__ import print_function, division

from os import path

import editdistance
import numpy as np
# import tensorflow as tf
from multiprocessing import Pool

"""
In this module contains classes to create a feature vector for word pairs.

* The Features class takes a list of feature components and converts and constructs and tensorflow variable
that merges all features into one vector.

* Some features will be extracted/calculated outside the tensorflow graph, e.g., edit distance. These
features you should be wrapped inside FeaturePlaceholderComponent before you can pass them to Features.

* Other features like WordEmbeddings are extracted inside the tensorflow graph.

"""


class Features(object):
  def __init__(self, feature_components, fix_features=False, use_infer_graph=False):
    import tensorflow as tf

    self._feature_components = feature_components
    features_infer = [fc.output_infer for fc in feature_components]
    features_infer = tf.concat(features_infer, 1, name='features_infer')
    if fix_features:
      features_infer = tf.stop_gradient(features_infer, name='fixed_features_infer')
      print('Fixing features.')
    if use_infer_graph:
      features = features_infer
    else:
      features = [fc.output for fc in feature_components]
      features = tf.concat(features, 1, name='features')
      if fix_features:
        features = tf.stop_gradient(features, name='fixed_features')
    self._features = features
    self._features_infer = features_infer

  def input_feeds(self, **inputs):
    input_feeds = {}
    for c in self._feature_components:
      new_feeds = c.input_feeds(**inputs)
      for k, v in new_feeds.items():
        if k in input_feeds:
          raise ValueError('Key already in feed dict.')
        else:
          input_feeds[k] = v
    return input_feeds

  @property
  def output(self):
    return self._features

  @property
  def output_infer(self):
    return self._features_infer

  def feature_names(self):
    return [c.name() for c in self._feature_components]

  def features_config_name(self):
    return '.'.join(self.feature_names())


class _EditDistanceHelper(object):
  def __init__(self, w_s):
    self.w_s = w_s

  def __call__(self, w_t):
    return editdistance.eval(self.w_s, w_t)


class EditDistance(object):
  def __init__(self, vocabulary_source, vocabulary_target, edit_distance_matrix_file, njobs=4):
    self._words_s = vocabulary_source
    self._words_t = vocabulary_target
    self._word2id_source = {w: i for i, w in enumerate(vocabulary_source)}
    self._word2id_target = {w: i for i, w in enumerate(vocabulary_target)}
    self._edit_distance_matrix_file = edit_distance_matrix_file
    self._edit_distance_matrix = None
    self._edit_distance_matrix = self.get_edit_distance_matrix(njobs=njobs)

  def get_edit_distance_matrix(self, njobs=4):
    if self._edit_distance_matrix is not None:
      return self._edit_distance_matrix

    edit_distance_matrix_file = self._edit_distance_matrix_file
    if path.exists(edit_distance_matrix_file):
      print('Loading editdistances...')
      res = np.load(edit_distance_matrix_file)
      print('Done.')
      return res

    pool = Pool(njobs)
    edit_distance_matrix = np.zeros((len(self._words_s), len(self._words_t)))
    for i, w_s in enumerate(self._words_s):
      if i % 100 == 0:
        print('\rCalculating edit distance matrix:', round(i / len(self._words_s) * 100, 2), '%', end='')
      # for j, w_t in enumerate(self._words_t):
      #   edit_distance_matrix[i, j] = editdistance.eval(w_s, w_t)

      edh = _EditDistanceHelper(w_s)
      tmp_lst = pool.map(edh, self._words_t)
      for j, w_t in enumerate(tmp_lst):
        edit_distance_matrix[i, j] = w_t

    np.save(edit_distance_matrix_file, edit_distance_matrix)
    pool.close()
    return edit_distance_matrix

  def extract_features(self, **raw_input):
    source_words, target_words = raw_input['source/word'], raw_input['target/word']
    num_examples = len(source_words)
    edit_distances = np.zeros(shape=(num_examples, 1))
    for i in xrange(num_examples):
      w_s = source_words[i]
      w_t = target_words[i]
      wid_s = self._word2id_source[w_s]
      wid_t = self._word2id_target[w_t]
      edit_distances[i] = self._edit_distance_matrix[wid_s, wid_t]
      #edit_distances[i] = editdistance.eval(w_s, w_t)
    return edit_distances

  def name(self):
    return 'ED'

  def feature_dim(self):
    return 1


class NormalizedEditDistance(object):
  def __init__(self, edit_distance):
    self._edit_distance = edit_distance

  def extract_features(self, **raw_input):
    source_words, target_words = raw_input['source/word'], raw_input['target/word']
    edit_distances = self._edit_distance.extract_features(**raw_input)
    for i, t in enumerate(zip(source_words, target_words)):
      av_len = (len(t[0]) + len(t[1])) / 2
      edit_distances[i] /= av_len
    return edit_distances

  def name(self):
    return 'ED_norm'

  def feature_dim(self):
    return 1


class EditDistanceRank(object):
  """Assumes you know all terms in advance => in case of MWU not really applicable """

  def __init__(self, vocabulary_source, vocabulary_target, edit_distance, edit_distance_rank_file):
    words_s = vocabulary_source
    words_t = vocabulary_target
    self._word2id_source = {w: i for i, w in enumerate(vocabulary_source)}
    self._word2id_target = {w: i for i, w in enumerate(vocabulary_target)}
    if path.exists(edit_distance_rank_file):
      with open(edit_distance_rank_file, 'rb') as f:
        arrays = np.load(f)
        self._rank_source2target = arrays['rank_source2target']
        self._rank_target2source = arrays['rank_target2source']
        return
    edit_distance_matrix_st = edit_distance.get_edit_distance_matrix()

    min_edit_indices_st = edit_distance_matrix_st.argsort(axis=1)
    edit_distance_matrix_ts = edit_distance_matrix_st.transpose()
    min_edit_indices_ts = edit_distance_matrix_ts.argsort(axis=1)
    print('  Calculating rank dictionary source to target.')
    num_source_words = len(words_s)
    num_target_words = len(words_t)

    rank_matrix_st = np.zeros(dtype=np.int16, shape=(num_source_words, num_target_words))
    for i in xrange(num_source_words):
      d_previous = -1
      rank_previous = 1
      for k in xrange(num_target_words):
        j = min_edit_indices_st[i, k]  # index target word
        rank = k + 1  # rank in range 1..N
        d = edit_distance_matrix_st[i, j]
        if d == d_previous:
          rank = rank_previous
        else:
          d_previous = d
          rank_previous = rank
        rank_matrix_st[i, j] = rank

    print('  Calculating rank dictionary target to source.')
    rank_matrix_ts = np.zeros(shape=(num_target_words, num_source_words), dtype=np.int16)
    for i in xrange(num_target_words):
      d_previous = -1
      rank_previous = 1
      for k in xrange(num_source_words):
        j = min_edit_indices_ts[i, k]  # index source word
        d = edit_distance_matrix_ts[i, j]
        rank = k + 1  # rank in range 1..N
        if d == d_previous:
          rank = rank_previous
        else:
          d_previous = d
          rank_previous = rank
        rank_matrix_ts[i, j] = rank
    self._rank_source2target = rank_matrix_st
    self._rank_target2source = rank_matrix_ts
    print('  Saving matrices.')
    with open(edit_distance_rank_file, 'wb') as f:
      np.savez(f,
               rank_source2target=self._rank_source2target,
               rank_target2source=self._rank_target2source)

  def extract_features(self, **raw_input):
    source_words, target_words = raw_input['source/word'], raw_input['target/word']
    num_examples = len(source_words)
    ranks = np.zeros(shape=(num_examples, 2))
    for i in xrange(num_examples):
      w_s = source_words[i]
      w_t = target_words[i]
      i_w_s, i_w_t = self._word2id_source[w_s], self._word2id_target[w_t]

      ranks[i, 0] = self._rank_source2target[(i_w_s, i_w_t)]
      ranks[i, 1] = self._rank_target2source[(i_w_t, i_w_s)]
    return ranks

  def name(self):
    return 'ED_rank'

  def feature_dim(self):
    return 2


class EditDistanceLogRank(object):
  def __init__(self, edit_distance_rank):
    self._edit_distance_rank = edit_distance_rank

  def extract_features(self, **raw_input):
    editdistance_ranks = self._edit_distance_rank.extract_features(**raw_input)
    return np.log10(editdistance_ranks)

  def name(self):
    return 'ED_log_rank'

  def feature_dim(self):
    return self._edit_distance_rank.feature_dim()


class FeaturePlaceholderComponent(object):
  def __init__(self, feature_extractor):
    import tensorflow as tf

    self._feature_extractor = feature_extractor
    dim = feature_extractor.feature_dim()
    name = feature_extractor.name()
    self._feature = tf.placeholder(
      dtype=tf.float32, shape=[None, dim], name=name)
    self._name = name

  def input_feeds(self, **features):
    return {self._feature: self._feature_extractor.extract_features(**features)}

  @property
  def output(self):
    return self._feature

  @property
  def output_infer(self):
    return self._feature

  def name(self):
    return self._name


class Word2Id(object):
  def __init__(self, vocabulary, scope=''):
    self._word2id = {w: i for i, w in enumerate(vocabulary)}
    self._scope = scope

  def extract_features(self, **raw_input):
    words = raw_input[self._scope + 'word']
    num_examples = len(words)
    wordids = np.zeros(shape=(num_examples,), dtype=np.int32)
    for i in xrange(num_examples):
      w = words[i]
      wordids[i] = self._word2id[w] if w in self._word2id else len(self._word2id)
    return wordids,


class CharLevelInputExtraction(object):
  def __init__(self, char_vocab, max_len, scope, time_major=True, word_delimiters=False, word_lens=True):
    self._max_len = max_len
    self._prefix = scope
    self._char2id = {c: i for i, c in enumerate(char_vocab)}
    self._time_major = time_major
    self._word_delimiters = word_delimiters
    self._word_lens = word_lens

  def extract_features(self, **kwargs):
    words = kwargs[self._prefix + 'word']
    word_lens = [len(w) for w in words]
    offset = 1 if self._word_delimiters else 0
    seq_len = self._max_len
    if self._time_major:
      chars = np.zeros(shape=(seq_len, len(words)))
      for j, w in enumerate(words):
        for i, c in enumerate(w):
          chars[i + offset, j] = self._char2id[c]
        if self._word_delimiters:
          chars[0, j] = self._char2id['<bow>']
          chars[len(w) + 1, j] = self._char2id['<eow>']
    else:  # batch major
      chars = np.zeros(shape=(len(words), seq_len))
      for i, w in enumerate(words):
        for j, c in enumerate(w):
          chars[i, j + offset] = self._char2id[c]
        if self._word_delimiters:
          chars[i, 0] = self._char2id['<bow>']
          chars[i, len(w) + 1] = self._char2id['<eow>']
    if self._word_lens:
      return chars, word_lens
    else:
      return chars


class CombineFeatureExtraction(object):
  def __init__(self, feature_extraction1, feature_extraction2):
    self._fe1 = feature_extraction1
    self._fe2 = feature_extraction2

  def extract_features(self, **kwargs):
    f1 = self._fe1.extract_features(**kwargs)
    f2 = self._fe2.extract_features(**kwargs)
    return f1 + f2
