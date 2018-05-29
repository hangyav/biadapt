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
from gensim.models import KeyedVectors

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')

flags.DEFINE_string('architecture', 'svhn_model', 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 500, 'Batch size for eval loop.')

flags.DEFINE_integer('new_size', 0, 'If > 0, resize image to this width/height.'
                                    'Needs to match size used for training.')

flags.DEFINE_integer('emb_size', 128,
                     'Size of the embeddings to learn.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/tmp/semisup',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('timeout', 1200,
                     'The maximum amount of time to wait between checkpoints. '
                     'If left as `None`, then the process will wait '
                     'indefinitely.')

flags.DEFINE_string('dictionaries', None,
                    'Dictionaries used for prerpocessing the input data.')

flags.DEFINE_string('w2v', None,
                    'Word2vec models, separated by ,')

flags.DEFINE_integer('word_embedding_dim', 300,
                     '')

def main(_):
    # Get dataset-related toolbox.
    dataset_tools = import_module('tools.' + FLAGS.dataset)
    architecture = getattr(semisup.architectures, FLAGS.architecture)

    nb_words=None
    if architecture == semisup.architectures.cnn_sentiment_model:
        with open(FLAGS.dictionaries, 'rb') as fin:
            tokenizer, encoder, nb_words, seq_len = pickle.load(fin)

        dataset_tools.IMAGE_SHAPE = [seq_len]
        dataset_tools.NUM_LABELS = len(encoder.classes_)

    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE

    test_images, test_labels = dataset_tools.get_data('test')



    graph = tf.Graph()
    with graph.as_default():

        # Set up input pipeline.
        image, label = tf.train.slice_input_producer([test_images, test_labels])
        images, labels = tf.train.batch(
            [image, label], batch_size=FLAGS.eval_batch_size)
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int64)

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
                w2v = KeyedVectors.load_word2vec_format(path, unicode_errors='ignore')
                dim = w2v.vector_size

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
            image_shape,
            test_in=images)

        # Add moving average variables.
        for var in tf.get_collection('moving_vars'):
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
        for var in slim.get_model_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

        # Get prediction tensor from semisup model.
        predictions = tf.argmax(model.test_logit, 1)

        # Accuracy metric for summaries.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })
        for name, value in names_to_values.items():
            tf.summary.scalar(name, value)

        # Run the actual evaluation loop.
        num_batches = math.ceil(len(test_labels) / float(FLAGS.eval_batch_size))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.logdir + '/train',
            logdir=FLAGS.logdir + '/eval',
            num_evals=num_batches,
            eval_op=[v for v in names_to_updates.values()],
            eval_interval_secs=FLAGS.eval_interval_secs,
            session_config=config,
            timeout=FLAGS.timeout
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
