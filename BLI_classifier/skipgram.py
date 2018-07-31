import tensorflow as tf
import component
from features import Word2Id


def skipgram_nce_loss(num_words, num_samples, counts, word_emb, context_word, name=''):
    word_emb_dim = word_emb.get_shape()[1].value
    prefix = name + '/skipgram_loss/' if name else 'skipgram_loss/'
    with tf.variable_scope(prefix):
        softmax_Wt = tf.get_variable('softmax_weights', shape=[num_words, word_emb_dim], initializer=tf.zeros_initializer())

        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                            true_classes=context_word,
                            num_true=1,
                            num_sampled=num_samples,
                            unique=True,
                            range_max=num_words,
                            distortion=0.75,
                            unigrams=counts)

        # Weights for labels: [batch_size x word_emb_dim]
        true_w = tf.nn.embedding_lookup(softmax_Wt, context_word)
        # [batch_size]
        true_logits = tf.reduce_sum(tf.multiply(word_emb, true_w), [1])

        # Weights for sampled ids: [num_sampled, word_emb_dim]
        sampled_w = tf.nn.embedding_lookup(softmax_Wt, sampled_ids)
        # [batch_size x num_sampled]
        sampled_logits = tf.matmul(word_emb,
                                   sampled_w,
                                   transpose_b=True)

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits), name='loss_positive_examples')
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sampled_logits, labels=tf.zeros_like(sampled_logits), name='loss_negative_examples')
        nce_loss_tensor = tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)
        batch_size = tf.shape(word_emb)[0]
        loss = nce_loss_tensor/tf.cast(batch_size, tf.float32)
        return loss


class SkipgramNCELoss(object):

    def __init__(self, words, num_samples, counts, name=''):
        self._num_words = len(words)
        self._words = words
        self._num_samples = num_samples
        self._counts = counts
        self._name = name

    def __call__(self, word_emb):
        num_words = self._num_words
        num_samples = self._num_samples
        counts = self._counts
        prefix = self._name + '/' if self._name else ''
        with tf.device('/cpu:0'):
            with tf.variable_scope(prefix + 'context'):
                context_word = tf.placeholder(dtype=tf.int64, shape=[None], name='word')
                context_word_ = tf.reshape(context_word, [-1, 1], name='word_')
                loss = skipgram_nce_loss(num_words, num_samples, counts, word_emb, context_word_)
                word2id_extractor = Word2Id(self._words, prefix + 'context/')
                return component.Component([context_word], loss, feature_extractor=word2id_extractor)
