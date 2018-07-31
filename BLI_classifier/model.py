import argparse

import tensorflow as tf

import wordencoding
from classifier import Classifier
import features
from io_ import embeddings_and_vocab, char_vocab, load_lexicon
from candidate_generation import RandomCandidateGenerator, CandidateGenerator
from training import Trainer, EpochLossLogger, BasicStatsLogger, Evaluation
import data
import pickle

# EDIT_DISTANCE_FILE = 'edit_distance.npy'
#LOG_DIR = 'logs/charLSTM_Embs'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_neg_samples', type=int, default=10)
  # parser.add_argument('--num_candidates', type=int, default=10)
  parser.add_argument('--num_cells', type=int, nargs='+', default=[128, 128])
  parser.add_argument('--num_hidden_layers', type=int, default=2)
  parser.add_argument('--num_epochs', type=int, default=300)
  parser.add_argument('--batch_size', type=int, default=10)
  parser.add_argument('--embedding_file', type=str, default='eacl_data/ennl.mono.dim=50.bin')
  # parser.add_argument('--bwesg_embedding_file', type=str, default='eacl_data/ennl.bwesg.dim=50.window=100.bin')
  parser.add_argument('--training_data', type=str, default='eacl_data/lex.filtered.train80-20.txt')
  parser.add_argument('--test_data', type=str, default='eacl_data/lex.filtered.test80-20.txt')
  parser.add_argument('--out_dir', type=str, default='logs/charLSTM_Embs')
  # parser.add_argument('--editdistance_file', type=str, default='edit_distance.npy')
  parser.add_argument('--candidates_file', type=str)
  args = parser.parse_args()

  num_neg_samples = args.num_neg_samples
  # num_candidates = args.num_candidates
  num_cells = args.num_cells
  num_hidden_layers = args.num_hidden_layers
  num_epochs = args.num_epochs
  training_data = args.training_data
  test_data = args.test_data
  batch_size = args.batch_size
  LOG_DIR = args.out_dir
  # EDIT_DISTANCE_FILE = args.editdistance_file
  candidates_file = args.candidates_file

  vocab_S, vocab_T, _, Embs_S, Embs_T, _ = embeddings_and_vocab(args.embedding_file)
  # _, _, _, multi_Embs_S, multi_Embs_T, _ = embeddings_and_vocab(args.bwesg_embedding_file)
  char_vocab_S, char_vocab_T = char_vocab(vocab_S, vocab_T)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as session:
    classifier = create_classifier(
      session,
      vocab_S, vocab_T, Embs_S, Embs_T, char_vocab_S, char_vocab_T,
      num_cells, num_hidden_layers,
      num_neg_samples, candidates_file)
    trainer = create_trainer(classifier, batch_size, num_epochs, training_data, test_data, LOG_DIR)
    trainer.train()


def create_trainer(classifier, batch_size, num_epochs, training_data, test_data, LOG_DIR):
  training_data_feeder = data.BilexDataFeeder(training_data, batch_size, shuffle=True)
  trainer = Trainer(classifier, num_epochs, training_data_feeder)

  test_data_feeder = data.BilexDataFeeder(test_data, batch_size, shuffle=True)
  training_lexicon = load_lexicon(training_data)
  trainer.add_command(EpochLossLogger(classifier, LOG_DIR))
  trainer.add_command(BasicStatsLogger(classifier, training_data_feeder, num_epochs, 10))
  trainer.add_command(Evaluation(classifier, test_data_feeder, training_lexicon, num_epochs, LOG_DIR, ['all']))
  return trainer


def create_classifier(
    session,
    vocab_S, vocab_T, Embs_S, Embs_T, char_vocab_S, char_vocab_T,
    num_cells, num_hidden_layers,
    num_neg_samples, candidates):
  feats = define_features(vocab_S, vocab_T, char_vocab_S, char_vocab_T, Embs_S, Embs_T, num_cells)
  negative_sampling = RandomCandidateGenerator(vocab_S, vocab_T, num_neg_samples)
  # edit_distance = features.EditDistance(vocab_S, vocab_T, EDIT_DISTANCE_FILE)
  # candidate_generation = CandidateGenerator(vocab_S, vocab_T, multi_Embs_S, multi_Embs_T, num_candidates, edit_distance)

  with open(candidates, 'rb') as inp:
    candidate_generation = pickle.load(inp)

  hidden_dims = num_hidden_layers * [int(feats.output.get_shape()[1].value)]
  classifier = Classifier(session, vocab_S, vocab_T, feats, negative_sampling, candidate_generation, hidden_dims)
  return classifier


def define_features(
    vocab_S, vocab_T, char_vocab_S, char_vocab_T, Embs_S, Embs_T, num_cells):
  maximum_length_S = max(len(w) for w in vocab_S)
  maximum_length_T = max(len(w) for w in vocab_T)
  maximum_length = max(maximum_length_S, maximum_length_T)
  char_features_encoder = wordencoding.BilingualRNNEncoding(char_vocab_S, char_vocab_T, num_cells)
  charlevel_features = char_features_encoder(maximum_length)

  wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_S, embeddings=Embs_S, scope='source')
  word_embs_S = wordlevel_features_encoder()
  wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_T, embeddings=Embs_T, scope='target')
  word_embs_T = wordlevel_features_encoder()
  return features.Features([charlevel_features, word_embs_S, word_embs_T])
  #return features.Features([word_embs_S, word_embs_T])

if __name__ == '__main__':
  main()
