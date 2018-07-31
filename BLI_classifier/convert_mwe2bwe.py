import argparse
import io_ as io
import numpy as np

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e1', '--embedding_file1', type=str)
  parser.add_argument('-l1', '--language1', type=str, default='en')
  parser.add_argument('-e2', '--embedding_file2', type=str)
  parser.add_argument('-l2', '--language2', type=str, default='nl')
  parser.add_argument('-o', '--output', type=str)
  args = parser.parse_args()
  return args

def main(args):
  print "Loading model: {}".format(args.embedding_file1)
  voc1, emb1 = io.load_embeddings(args.embedding_file1, binary=False)
  print "Loading model: {}".format(args.embedding_file2)
  voc2, emb2 = io.load_embeddings(args.embedding_file2, binary=False)

  print 'Converting...'
  voc = ['{}_{}'.format(v.encode('utf-8'), args.language1).decode('utf-8') for v in voc1] + ['{}_{}'.format(v.encode('utf-8'), args.language2).decode('utf-8') for v in voc2]
  emb = np.concatenate((emb1, emb2), axis=0)

  print "Saving BWE to: {}".format(args.output)
  io.save_embeddings(args.output, emb, voc)


if __name__ == '__main__':
  args = get_args()
  main(args)
