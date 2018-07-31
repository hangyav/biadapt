import argparse

import tensorflow as tf

import wordencoding
from classifier import SemiSupClassifier
import features
from io_ import embeddings_and_vocab, char_vocab, load_lexicon
from candidate_generation import RandomCandidateGenerator, CandidateGenerator, MixedCandidateGenerator
from training import SemiSupTrainer, SemiSupEpochLossLogger, SemiSupBasicStatsLogger, Evaluation
import data
import pickle


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_neg_samples', type=int, default=10)
  parser.add_argument('--num_cells', type=int, nargs='+', default=[128, 128])
  parser.add_argument('--num_hidden_layers', type=int, default=2)
  parser.add_argument('--num_epochs', type=int, default=300)
  parser.add_argument('--batch_size', type=int, default=10)
  parser.add_argument('--embedding_file', type=str, default='eacl_data/ennl.mono.dim=50.bin')
  parser.add_argument('--training_data', type=str, default='eacl_data/lex.filtered.train80-20.txt')
  parser.add_argument('--unlabeled_data', type=str, default='eacl_data/lex.filtered.train80-20.tune.txt')
  parser.add_argument('--test_data', type=str, default='eacl_data/lex.filtered.test80-20.txt')
  parser.add_argument('--out_dir', type=str, default='logs/charLSTM_Embs')
  parser.add_argument('--test_candidates_file', type=str)
  parser.add_argument('--unlabeled_candidates_file', type=str)
  parser.add_argument('--walker_weight', type=float, default=1.0)
  parser.add_argument('--visit_weight', type=float, default=1.0)
  parser.add_argument('--logit_weight', type=float, default=1.0)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--add_negative_unlabeled', type=int, default=0)
  parser.add_argument('--reversed_unlabeled_candidates', type=int, default=0)
  args = parser.parse_args()

  num_neg_samples = args.num_neg_samples
  num_cells = args.num_cells
  num_hidden_layers = args.num_hidden_layers
  num_epochs = args.num_epochs
  training_data = args.training_data
  unlabeled_data = args.unlabeled_data
  test_data = args.test_data
  batch_size = args.batch_size
  LOG_DIR = args.out_dir
  test_candidates_file = args.test_candidates_file
  unlabeled_candidates_file = args.unlabeled_candidates_file
  walker_weight = args.walker_weight
  visit_weight = args.visit_weight
  logit_weight = args.logit_weight
  learning_rate = args.learning_rate
  add_negative_unlabeled = args.add_negative_unlabeled == 1
  unlabeled_source2target = args.reversed_unlabeled_candidates == 0


  vocab_S, vocab_T, _, Embs_S, Embs_T, _ = embeddings_and_vocab(args.embedding_file)
  char_vocab_S, char_vocab_T = char_vocab(vocab_S, vocab_T)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as session:
    classifier = create_classifier(
      session,
      vocab_S, vocab_T, Embs_S, Embs_T, char_vocab_S, char_vocab_T,
      num_cells, num_hidden_layers,
      num_neg_samples, batch_size, test_candidates_file, unlabeled_candidates_file,
      walker_weight=walker_weight, visit_weight=visit_weight, logit_weight=logit_weight, lr=learning_rate,
      add_negative_unlabeled=add_negative_unlabeled, unlabeled_source2target=unlabeled_source2target)
    trainer = create_trainer(classifier, batch_size, num_epochs, training_data, unlabeled_data, test_data, LOG_DIR)
    trainer.train()


def create_trainer(classifier, batch_size, num_epochs, training_data, unlabeled_data, test_data, LOG_DIR):
  training_data_feeder = data.BilexDataFeeder(training_data, batch_size, shuffle=True)
  unlabeled_data_feeder = data.BilexDataFeeder(unlabeled_data, batch_size, shuffle=True)
  trainer = SemiSupTrainer(classifier, num_epochs, training_data_feeder, unlabeled_data_feeder)

  test_data_feeder = data.BilexDataFeeder(test_data, batch_size, shuffle=True)
  training_lexicon = load_lexicon(training_data)
  trainer.add_command(SemiSupEpochLossLogger(classifier, LOG_DIR))
  trainer.add_command(SemiSupBasicStatsLogger(classifier, training_data_feeder, num_epochs, 10))
  trainer.add_command(Evaluation(classifier, test_data_feeder, training_lexicon, num_epochs, LOG_DIR, ['all']))
  return trainer


def create_classifier(
    session,
    vocab_S, vocab_T, Embs_S, Embs_T, char_vocab_S, char_vocab_T,
    num_cells, num_hidden_layers, num_neg_samples, batch_size,
    test_candidates_file, unlabeled_candidates_file,
    walker_weight, visit_weight, logit_weight, lr,
    add_negative_unlabeled=False, unlabeled_source2target=True):

  labeled_feats, unlabeled_feats = define_features(vocab_S, vocab_T, char_vocab_S, char_vocab_T, Embs_S, Embs_T, num_cells)
  negative_sampling = RandomCandidateGenerator(vocab_S, vocab_T, num_neg_samples)

  with open(test_candidates_file, 'rb') as inp:
    test_candidate_generation = pickle.load(inp)

  with open(unlabeled_candidates_file, 'rb') as inp:
    unlabeled_candidate_generation = pickle.load(inp)

  if add_negative_unlabeled:
    unlabeled_candidate_generation = MixedCandidateGenerator(unlabeled_candidate_generation,
                                                             RandomCandidateGenerator(vocab_S, vocab_T, unlabeled_candidate_generation.num_candidates))

  hidden_dims = num_hidden_layers * [int(labeled_feats.output.get_shape()[1].value)]
  classifier = SemiSupClassifier(session, vocab_S, vocab_T, labeled_feats, unlabeled_feats, negative_sampling,
                                 test_candidate_generation, unlabeled_candidate_generation,
                                 batch_size=(batch_size + batch_size*num_neg_samples*2), hidden_dims=hidden_dims,
                                 walker_weight=walker_weight, visit_weight=visit_weight, logit_weight=logit_weight, lr=lr,
                                 unlabeled_source2target=unlabeled_source2target)
  return classifier


def define_features(
    vocab_S, vocab_T, char_vocab_S, char_vocab_T, Embs_S, Embs_T, num_cells):
  maximum_length_S = max(len(w) for w in vocab_S)
  maximum_length_T = max(len(w) for w in vocab_T)
  maximum_length = max(maximum_length_S, maximum_length_T)
  char_features_encoder = wordencoding.BilingualRNNEncoding(char_vocab_S, char_vocab_T, num_cells)
  labeled_charlevel_features = char_features_encoder(maximum_length)
  unlabeled_charlevel_features = char_features_encoder(maximum_length, reuse=True)

  wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_S, embeddings=Embs_S, scope='source')
  labeled_word_embs_S = wordlevel_features_encoder()
  unlabeled_word_embs_S = wordlevel_features_encoder()
  wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_T, embeddings=Embs_T, scope='target')
  labeled_word_embs_T = wordlevel_features_encoder()
  unlabeled_word_embs_T = wordlevel_features_encoder()
  return [features.Features([labeled_charlevel_features, labeled_word_embs_S, labeled_word_embs_T]),
          features.Features([unlabeled_charlevel_features, unlabeled_word_embs_S, unlabeled_word_embs_T])]

if __name__ == '__main__':
  main()
