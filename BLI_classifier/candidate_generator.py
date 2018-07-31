import argparse

import features
from io_ import embeddings_and_vocab
from candidate_generation import CandidateGenerator
import pickle
import os



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_candidates', type=int, default=10)
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--bwesg_embedding_file', type=str, default='eacl_data/ennl.bwesg.dim=50.window=100.bin')
  parser.add_argument('--output', type=str, default='logs/charLSTM_Embs')
  parser.add_argument('--editdistance_file', type=str, default='edit_distance.npy')
  args = parser.parse_args()

  num_candidates = args.num_candidates
  output = args.output
  EDIT_DISTANCE_FILE = args.editdistance_file
  if os.path.exists(output) and os.path.exists(EDIT_DISTANCE_FILE):
    print '{} exists!'.format(output)
    return

  vocab_S, vocab_T, _, multi_Embs_S, multi_Embs_T, _ = embeddings_and_vocab(args.bwesg_embedding_file)
  candidates = create_candidates(vocab_S, vocab_T, multi_Embs_S, multi_Embs_T, num_candidates, EDIT_DISTANCE_FILE, args.threads)

  print 'Saving candidates to: {}'.format(output)
  with open(output, 'wb') as output:
    pickle.dump(candidates, output)


def create_candidates(vocab_S, vocab_T, multi_Embs_S, multi_Embs_T,  num_candidates, EDIT_DISTANCE_FILE, njobs):
  edit_distance = features.EditDistance(vocab_S, vocab_T, EDIT_DISTANCE_FILE, njobs=njobs)
  candidate_generation = CandidateGenerator(vocab_S, vocab_T, multi_Embs_S, multi_Embs_T, num_candidates, edit_distance, njobs=njobs)
  return candidate_generation


if __name__ == '__main__':
  main()
