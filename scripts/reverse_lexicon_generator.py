
import argparse
from gensim.models import KeyedVectors
import csv
import pandas as pd
from tqdm import tqdm
import logging


def get_most_similar(word, f, t, topn=100):
    if word in f:
        return t.similar_by_vector(f[word], topn=topn)
    return list()


def main():
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_candidates', type=int, default=3)
  parser.add_argument("--from_vectors", type=str, required=True)
  parser.add_argument("--to_vectors", type=str, required=True)
  parser.add_argument("--from_lang", default='en', type=str)
  parser.add_argument("--to_lang", default='nl', type=str)
  parser.add_argument('--output', type=str, required=True)
  parser.add_argument('--input', type=str, required=True)
  args = parser.parse_args()

  num_candidates = args.num_candidates
  output = args.output
  from_lang = args.from_lang
  to_lang = args.to_lang

  fw2v = KeyedVectors.load_word2vec_format(args.from_vectors, binary=False)
  tw2v = KeyedVectors.load_word2vec_format(args.to_vectors, binary=False)

  data = pd.read_csv(args.input, sep='\t', quoting=csv.QUOTE_NONE)

  with open(output, 'w') as fout:
    for i,r in tqdm(data.iterrows(), total=data.shape[0]):
      for w,v in get_most_similar(r[from_lang], fw2v, tw2v, topn=num_candidates):
        fout.write('{}\t{}\n'.format(w, r[from_lang]))


if __name__ == '__main__':
  main()
