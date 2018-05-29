#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised eval module.

This script defines the evaluation loop that works with the training loop
from train.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from functools import partial
from importlib import import_module

import semisup
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import pickle
import numpy as np
from tensorflow.python.lib.io import file_io
from io import BytesIO

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset in numpy format.')

flags.DEFINE_string('architecture', 'cnn_sentiment_model', 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for eval loop.')

flags.DEFINE_integer('emb_size', 1000,
                     'Size of the embeddings to learn.')

flags.DEFINE_string('dictionaries', None,
                    'Dictionaries used for prerpocessing the input data.')

flags.DEFINE_string('w2v', None,
                    'Word2vec models, separated by ,')

flags.DEFINE_integer('word_embedding_dim', 300,
                     '')

flags.DEFINE_string('model_path', None,
                    'Model path')

flags.DEFINE_string('output', None,
                    'output')

flags.DEFINE_integer('new_size', 0,
                     'If > 0, resize image to this width/height.')

def main(_):
    architecture = getattr(semisup.architectures, FLAGS.architecture)

    if FLAGS.dataset.startswith('gs'):
        data = np.load(BytesIO(file_io.read_file_to_string(FLAGS.dataset)))['data']
    else:
        data = np.load(FLAGS.dataset)['data']

    tokenizer = None
    encoder = None
    nb_words = None
    seq_len = None
    if architecture == semisup.architectures.cnn_sentiment_model:
        if FLAGS.dictionaries.startswith('gs'):
            with BytesIO(file_io.read_file_to_string(FLAGS.ddictionaries)) as fin:
                tokenizer, encoder, nb_words, seq_len = pickle.load(fin)
        else:
            with open(FLAGS.dictionaries, 'rb') as fin:
                tokenizer, encoder, nb_words, seq_len = pickle.load(fin)

        num_labels = len(encoder.classes_)
        image_shape = [seq_len]

    else:
        raise ValueError('Only cnn_sentiment model is supported in the moment!')



    graph = tf.Graph()
    with graph.as_default():

        # Reshape if necessary.
        if FLAGS.new_size > 0:
            new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
        else:
            new_shape = None

        # Create function that defines the network.
        if architecture == semisup.architectures.cnn_sentiment_model:

            dim = None
            if FLAGS.w2v is not None:
                path = FLAGS.w2v.split(',')[0]
                tf.logging.info('Loading word2vec: {}'.format(path))
                with open(path, 'r') as fin:
                    dim = int(fin.readline().split()[1])

            if dim is None:
                dim = FLAGS.word_embedding_dim

            model_function = partial(
                architecture,
                nb_words=nb_words,
                embedding_dim=dim,
                static_embedding=1,
                embedding_weights=None,
                ############################################################
                img_shape=image_shape,
                emb_size=FLAGS.emb_size)
        else:
            model_function = partial(
                architecture,
                is_training=False,
                new_shape=new_shape,
                img_shape=image_shape,
                augmentation_function=None,
                image_summary=False,
                emb_size=FLAGS.emb_size)


        # Set up semisup model.
        model = semisup.SemisupModel(
            model_function,
            num_labels,
            image_shape)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))

            idx = 0
            batch_size = FLAGS.eval_batch_size
            num_elements = data.shape[0]
            res = np.zeros((num_elements, FLAGS.emb_size))

            while idx < num_elements:
                r_idx = min(idx+batch_size, num_elements)

                tf.logging.info('Running: {}-{}/{}'.format(idx, r_idx, num_elements))

                batch = data[idx:r_idx]
                res[idx:r_idx] = model.calc_embedding(batch, model.test_emb)

                idx += batch_size

            tf.logging.info('Saving results to: {}'.format(FLAGS.output))
            np.save(FLAGS.output, res)





if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
