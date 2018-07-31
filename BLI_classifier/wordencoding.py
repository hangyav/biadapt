import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn import _reverse_seq


from component import Component
from features import Word2Id, CharLevelInputExtraction, CombineFeatureExtraction

from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn, seq2seq


def _get_last_state_dyn(max_norm, lengths, outputs):
  outputs_shape = tf.shape(outputs)
  outputs = tf.reshape(outputs, [-1, outputs.get_shape()[2].value])
  indices = lengths - 1
  shift = tf.range(outputs_shape[0]) * outputs_shape[1]
  indices = indices + shift
  last = tf.nn.embedding_lookup(outputs, indices)
  if max_norm:
    last = clip_by_norm(last, max_norm)
  return last


def _get_last_state(max_norm, char_lens, outputs):
  # A bit of a hack to get the last real output state.
  outputs = _reverse_seq(outputs, tf.cast(char_lens, dtype=tf.int64))
  output = outputs[0]

  if max_norm:
    output = clip_by_norm(output, max_norm)
  return output


def _get_cell(input_dim, hidden_dims, dropout):
  if type(hidden_dims) == list and len(hidden_dims) > 1:
    # multi layer rnn
    dims = [input_dim] + hidden_dims
    cells = []
    for i in xrange(len(dims) - 2):
      cell = tf.contrib.rnn.LSTMCell(num_units=dims[i + 1], cell_clip=3.,
                                  state_is_tuple=True)
      if dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=dropout)
      cells.append(cell)

    last_cell = tf.contrib.rnn.LSTMCell(num_units=dims[-1], cell_clip=3.,
                                     state_is_tuple=True)
    if dropout:
      last_cell = tf.contrib.rnn.DropoutWrapper(last_cell, input_keep_prob=1, output_keep_prob=dropout)
    cells.append(last_cell)
    return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

  else:
    # single layer rnn
    if type(hidden_dims) == list:
      hidden_dim = hidden_dims[0]
    else:
      hidden_dim = hidden_dims
    cell = tf.contrib.rnn.LSTMCell(num_units=hidden_dim,
                                state_is_tuple=True)
    if dropout:
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1., output_keep_prob=dropout)
    return cell


def clip_by_norm(t, clip_norm, name=None):
  with ops.op_scope([t, clip_norm], name, "clip_by_norm") as name:
    t = ops.convert_to_tensor(t, name="t")

    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    l2norm_inv = math_ops.rsqrt(
      math_ops.reduce_sum(t * t, reduction_indices=1, keep_dims=True))
    tclip = array_ops.identity(t * clip_norm * math_ops.minimum(
      l2norm_inv, array_ops.constant(1.0 / clip_norm)), name=name)
  return tclip


class BilingualRNNEncoding(object):
  def __init__(self, char_vocab_S, char_vocab_T,
               num_cells, max_norm=None, word_delimiters=True, dropout=0.5, name=''):
    self._char_vocab_source = char_vocab_S
    self._char_vocab_target = char_vocab_T

    self._variable_scope = name + '/' if name else ''
    with tf.variable_scope(name) as outer_scope:
      self.Wchar_s = self._one_hot_embs(char_vocab_S, 'source/char_embs', outer_scope)
      self.Wchar_t = self._one_hot_embs(char_vocab_T, 'target/char_embs', outer_scope)

      with tf.variable_scope('rnn_cell') as s:
        num_chars = len(char_vocab_S) + len(char_vocab_T)
        self.char_rnn_cell_infer = _get_cell(num_chars, num_cells, None)
        if dropout:
          s.reuse_variables()
          self.char_rnn_cell_train = _get_cell(num_chars, num_cells, dropout)
        else:
          self.char_rnn_cell_train = self.char_rnn_cell_infer
      self.max_norm = max_norm
      self._word_delimiters = word_delimiters
      self._dropout = dropout

  def _one_hot_embs(self, char_vocab, name, scope):
    with tf.variable_scope(scope):
      num_chars = len(char_vocab)
      identity_matrix = np.identity(num_chars, dtype=np.float32)
      Wchar = tf.get_variable(name, initializer=identity_matrix, trainable=False)
      return Wchar

  def __call__(self, max_len, reuse=False, name=''):
    component_scope = self._variable_scope + name
    with tf.variable_scope(component_scope) as scope:
      if reuse:
        scope.reuse_variables()
      max_len = max_len + 2 if self._word_delimiters else max_len
      word_lens_source = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_lens')
      word_lens_target = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_target')

      chars_p_s = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='source/char_sequence%i' % max_len)
      chars_p_t = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='target/char_sequence%i' % max_len)

      chars_s = tf.nn.embedding_lookup(self.Wchar_s, chars_p_s)
      chars_t = tf.nn.embedding_lookup(self.Wchar_t, chars_p_t)
      word_lens = tf.maximum(word_lens_source, word_lens_target)
      chars = tf.concat([chars_s, chars_t], 2)
      #output_infer, _ = tf.nn.dynamic_rnn(self.char_rnn_cell_infer, chars, sequence_length=tf.cast(word_lens, dtype=tf.int64))

      chars = [tf.squeeze(j, [0]) for j in tf.split(chars, max_len, 0)]
      outputs_infer, _ = tf.contrib.rnn.static_rnn(self.char_rnn_cell_infer, chars,
                                dtype=tf.float32, sequence_length=tf.cast(word_lens, dtype=tf.int64))
      scope.reuse_variables()
      outputs, _ = tf.contrib.rnn.static_rnn(
        self.char_rnn_cell_train, chars,
        dtype=tf.float32, sequence_length=tf.cast(word_lens_source, dtype=tf.int64))
      output = _get_last_state(self.max_norm, word_lens_source, outputs)
      output_infer = _get_last_state(self.max_norm, word_lens_source, outputs_infer)
      inputs = [
        chars_p_s, word_lens_source,
        chars_p_t, word_lens_target
      ]

      char_feature_extractor1 = CharLevelInputExtraction(self._char_vocab_source, max_len, component_scope + 'source/')
      char_feature_extractor2 = CharLevelInputExtraction(self._char_vocab_target, max_len, component_scope + 'target/')
      char_feature_extractor = CombineFeatureExtraction(char_feature_extractor1, char_feature_extractor2)
      return Component(inputs, output,
                       output_infer=output_infer,
                       feature_extractor=char_feature_extractor,
                       name='c_rnn_joint')


class BilingualAttentionRNNEncoding(object):
  def __init__(self, char_vocab_source, char_vocab_target,
               hidden_dims, max_norm=None, word_delimiters=True, dropout=0.5, name=''):
    self._char_vocab_source = char_vocab_source
    self._char_vocab_target = char_vocab_target

    self._variable_scope = name + '/' if name else ''
    with tf.variable_scope(name) as outer_scope:
      self.Wchar_s = self._one_hot_embs(char_vocab_source, 'source/char_embs', outer_scope)
      self.Wchar_t = self._one_hot_embs(char_vocab_target, 'target/char_embs', outer_scope)

      with tf.variable_scope('rnn_cell') as s:
        num_chars = len(char_vocab_source) + len(char_vocab_target)
        self.char_rnn_cell_infer = _get_cell(num_chars, hidden_dims, None)
        if dropout:
          s.reuse_variables()
          self.char_rnn_cell_train = _get_cell(num_chars, hidden_dims, dropout)
        else:
          self.char_rnn_cell_train = self.char_rnn_cell_infer
      self.max_norm = max_norm
      self._word_delimiters = word_delimiters
      self._dropout = dropout

  def _one_hot_embs(self, char_vocab, name, scope):
    with tf.variable_scope(scope):
      num_chars = len(char_vocab)
      identity_matrix = np.identity(num_chars, dtype=np.float32)
      Wchar = tf.get_variable(name, initializer=identity_matrix, trainable=False)
      return Wchar

  def __call__(self, name, max_len, reuse=False):
    component_scope = self._variable_scope + name
    with tf.variable_scope(component_scope) as scope:
      if reuse:
        scope.reuse_variables()
      max_len = max_len + 2 if self._word_delimiters else max_len
      word_lens_source = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_lens')
      word_lens_target = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_target')

      chars_p_s = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='source/char_sequence%i' % max_len)
      chars_p_t = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='target/char_sequence%i' % max_len)

      chars_s = tf.nn.embedding_lookup(self.Wchar_s, chars_p_s)
      chars_s = tf.transpose(chars_s, [1, 0, 2])
      chars_t = tf.nn.embedding_lookup(self.Wchar_t, chars_p_t)
      chars_t = tf.transpose(chars_t, [1, 0, 2])
      chars = tf.concat([chars_s, chars_t], 2)


      enc_output_infer, enc_state_infer = tf.nn.dynamic_rnn(self.char_rnn_cell_infer, chars,  dtype=tf.float32,
                                                            sequence_length=tf.cast(word_lens_source, dtype=tf.int64),
                                                            swap_memory=True, scope='encoder')
      attn_keys, attn_values, attn_score_fn, attn_construct_fn = attention_decoder_fn.prepare_attention(enc_output_infer, 'luong', self.char_rnn_cell_infer.output_size)
      dec_fn_inf = attention_decoder_fn.attention_decoder_fn_train(
        enc_state_infer, attn_keys, attn_values, attn_score_fn, attn_construct_fn)
      outputs_infer, _, _ = seq2seq.dynamic_rnn_decoder(self.char_rnn_cell_infer, dec_fn_inf,
                                                     inputs=chars, sequence_length=word_lens_target, swap_memory=True,
                                                     scope='decoder')

      scope.reuse_variables()
      enc_output, enc_state = tf.nn.dynamic_rnn(self.char_rnn_cell_train, chars, dtype=tf.float32,
                                                            sequence_length=tf.cast(word_lens_source, dtype=tf.int64),
                                                            swap_memory=True, scope='encoder')
      attn_keys, attn_values, attn_score_fn, attn_construct_fn = attention_decoder_fn.prepare_attention(
        enc_output, 'luong', self.char_rnn_cell_infer.output_size)
      dec_fn = attention_decoder_fn.attention_decoder_fn_train(
        enc_state, attn_keys, attn_values, attn_score_fn, attn_construct_fn)
      outputs, _, _ = seq2seq.dynamic_rnn_decoder(self.char_rnn_cell_train, dec_fn, inputs=chars,
                                              sequence_length=word_lens_target, swap_memory=True,
                                              scope='decoder')

      output = _get_last_state_dyn(self.max_norm, word_lens_target, outputs)
      output_infer = _get_last_state_dyn(self.max_norm, word_lens_target, outputs_infer)
      inputs = [
        chars_p_s, word_lens_source,
        chars_p_t, word_lens_target
      ]

      char_feature_extractor1 = CharLevelInputExtraction(self._char_vocab_source, max_len, component_scope + 'source/')
      char_feature_extractor2 = CharLevelInputExtraction(self._char_vocab_target, max_len, component_scope + 'target/')
      char_feature_extractor = CombineFeatureExtraction(char_feature_extractor1, char_feature_extractor2)
      return Component(inputs, output,
                       output_infer=output_infer,
                       feature_extractor=char_feature_extractor,
                       name='c_rnn_joint')


class BilingualBidirRNNEncoding(object):
  def __init__(self, char_vocab_source, char_vocab_target,
               hidden_dims, max_norm=None, word_delimiters=True, dropout=0.5, name=''):
    self._char_vocab_source = char_vocab_source
    self._char_vocab_target = char_vocab_target

    num_chars_source = len(char_vocab_source)
    num_chars_target = len(char_vocab_target)
    num_chars = num_chars_source + num_chars_target
    self._variable_scope = name + '/' if name else ''
    with tf.variable_scope(name) as outer_scope:
      identity_matrix = np.identity(num_chars_source, dtype=np.float32)
      self.Wchar_source = tf.get_variable('source/char_embs', initializer=identity_matrix, trainable=False)
      identity_matrix = np.identity(num_chars_target, dtype=np.float32)
      self.Wchar_target = tf.get_variable('target/char_embs', initializer=identity_matrix, trainable=False)

      hidden_dims = hidden_dims if type(hidden_dims) == list else [hidden_dims]

      with tf.variable_scope('rnn_cell') as scope:
        self.rnn_cell_fw_infer = _get_cell(num_chars, hidden_dims[0], None)
        self.rnn_cell_bw_infer = self.rnn_cell_fw_infer
        self.rnn_cell_other_layers_infer = _get_cell(hidden_dims[0], hidden_dims[1:], None) \
          if hidden_dims[1:] else None
        if dropout:
          scope.reuse_variables()
          self.rnn_cell_fw_train = _get_cell(num_chars, hidden_dims[0], dropout)
          self.rnn_cell_bw_train = self.rnn_cell_fw_train
          self.rnn_cell_other_layers_train = _get_cell(hidden_dims[0], hidden_dims[1:], dropout) \
            if hidden_dims[1:] else None
      self.max_norm = max_norm
      self._word_delimiters = word_delimiters
      self._dropout = dropout
      print(outer_scope.name)
      vars = tf.get_collection(tf.GraphKeys.VARIABLES)
      print(vars)

  def __call__(self, name, max_len, reuse=False):
    component_scope = self._variable_scope + name
    with tf.variable_scope(component_scope) as scope:
      if reuse:
        scope.reuse_variables()
      max_len = max_len + 2 if self._word_delimiters else max_len
      word_lens_source = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_lens')
      word_lens_target = tf.placeholder(dtype=tf.int32, shape=[None], name='source/word_target')

      chars_p_s = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='source/char_sequence%i' % max_len)
      chars_p_t = tf.placeholder(dtype=tf.int32, shape=[max_len, None],
                                 name='target/char_sequence%i' % max_len)

      chars_s = tf.nn.embedding_lookup(self.Wchar_source, chars_p_s)
      chars_t = tf.nn.embedding_lookup(self.Wchar_target, chars_p_t)

      chars = tf.concat(2, [chars_s, chars_t])
      chars = [tf.squeeze(j, [0]) for j in tf.split(0, chars.get_shape()[0], chars)]

      word_lens = tf.maximum(word_lens_source, word_lens_target)
      outputs_infer, _, _ = tf.contrib.static_bidirectional_rnn(self.rnn_cell_fw_train, self.rnn_cell_bw_infer, chars,
                                                 dtype=tf.float32, sequence_length=tf.cast(word_lens, dtype=tf.int64))
      if self.rnn_cell_other_layers_infer:
        outputs_infer, _ = tf.contrib.static_rnn(self.rnn_cell_other_layers_infer, chars,
                                     dtype=tf.float32, sequence_length=tf.cast(word_lens, dtype=tf.int64))
      scope.reuse_variables()
      outputs, _, _ = tf.contrib.static_bidirectional_rnn(self.rnn_cell_fw_train, self.rnn_cell_fw_train, chars,
                                           dtype=tf.float32, sequence_length=tf.cast(word_lens, dtype=tf.int64))
      if self.rnn_cell_other_layers_train:
        outputs, _ = tf.contrib.static_rnn(self.rnn_cell_other_layers_train, chars,
                               dtype=tf.float32, sequence_length=tf.cast(word_lens, dtype=tf.int64))
      output = _get_last_state(self.max_norm, word_lens_source, outputs)
      output_infer = _get_last_state(self.max_norm, word_lens_source, outputs_infer)
      inputs = [
        chars_p_s, word_lens_source,
        chars_p_t, word_lens_target
      ]

      char_feature_extractor1 = CharLevelInputExtraction(self._char_vocab_source, max_len,
                                                         component_scope + 'source/')
      char_feature_extractor2 = CharLevelInputExtraction(self._char_vocab_target, max_len,
                                                         component_scope + 'target/')
      char_feature_extractor = CombineFeatureExtraction(char_feature_extractor1, char_feature_extractor2)
      return Component(inputs, output,
                       output_infer=output_infer,
                       feature_extractor=char_feature_extractor,
                       name='c_bidir_rnn_joint')


class WordLevelEncoding(object):
  def __init__(self, vocabulary, scope='', shape=None, embeddings=None, dropout=None, trainable=False):
    self._vocabulary = vocabulary
    self._vocab_size = len(vocabulary)
    self._emb_size = embeddings.shape[1]
    with tf.variable_scope(scope):
      if embeddings is not None:
        embeddings = tf.concat([embeddings, np.ones((1,self._emb_size), dtype=np.float32)], axis=0)
        self._embeddings = tf.Variable(
          embeddings, name='embeddings', trainable=trainable, dtype=tf.float32)
      else:
        initial = tf.truncated_normal(shape, stddev=0.1)
        self._embeddings = tf.Variable(initial, name='embeddings')
    self._dropout = dropout
    self._parameter_scope = scope + '/' if scope else scope

  def __call__(self, component_scope=''):
    scope = self._parameter_scope + component_scope
    with tf.variable_scope(scope):
      words = tf.placeholder(dtype=tf.int32, shape=[None], name='wordidx')
      output = tf.nn.embedding_lookup(self._embeddings, words)
      output_infer = output

      if self._dropout:
        output = tf.nn.dropout(output, self._dropout)

      word2id_extractor = Word2Id(self._vocabulary, scope)
      inputs = [words]
      return Component(inputs, output, output_infer=output_infer,
                       feature_extractor=word2id_extractor, name='w_emb')
