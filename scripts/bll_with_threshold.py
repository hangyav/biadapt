#!/usr/bin/env python

import logging
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import csv

def get_options():
    parser = OptionParser()
    parser.add_option("-e", "--test", dest="test") # test lexicon from heyman
    parser.add_option("-o", "--output", dest="out")
    parser.add_option("--fv", "--from_vectors", dest="from_vectors")
    parser.add_option("--tv", "--to_vectors", dest="to_vectors")
    parser.add_option("-f", "--from", dest="from_lang", default='en')
    parser.add_option("-t", "--to", dest="to_lang", default='nl')
    parser.add_option("--topn", dest="topn", default=10, type=int)
    parser.add_option("--th_from", dest="th_from", default=0.6, type=float)
    parser.add_option("--th_to", dest="th_to", default=1.0, type=float)
    parser.add_option("--th_step", dest="th_step", default=0.001, type=float)

    (options, args) = parser.parse_args()
    return options

def get_most_similar(word, f, t, topn=100):
    if word in f:
        tmp = t.similar_by_vector(f[word], topn=topn)
        maxv = max([v for w,v in tmp])
        return [(w, v/maxv) for w,v in tmp]
    # return t.similar_by_vector(np.zeros(f.vector_size))
    return list()


def eval_with_th(source, gold, pred, th):
    ground_truth = zip(source, gold)
    translation_pairs = set()

    for i in range(len(gold)):
        s = source[i]
        for p in [w for w,v in pred[i] if v >= th]:
            translation_pairs.add((s, p))

    return eval_translations(ground_truth, translation_pairs)[2]

def eval_translations(groundtruth, translations):
    groundtruth_set = set(groundtruth)
    true_positives = groundtruth_set & translations

    recall = (len(true_positives) / len(groundtruth_set))

    precision = (len(true_positives) / len(translations)) if translations else 0.0

    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    return precision, recall, f1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    options = get_options()

    
    data = pd.read_csv(options.test, sep='\t', quoting=csv.QUOTE_NONE)
    fw2v = KeyedVectors.load_word2vec_format(options.from_vectors, binary=False)
    if options.to_vectors is None or options.from_vectors == options.to_vectors:
        tw2v = fw2v
    else:
        tw2v = KeyedVectors.load_word2vec_format(options.to_vectors, binary=False)

    data['contains'] = data[options.from_lang].apply(fw2v.__contains__)
    data['pred'] = [get_most_similar(w, fw2v, tw2v, topn=options.topn) for w in tqdm(data[options.from_lang])]

    print('OOV: {}'.format(1.0 - float(len(data[data.contains]))/len(data)))
    print('Contained word accuracy: {}'.format(accuracy_score(data[data.contains][options.to_lang], [v[0][0] for v in data[data.contains].pred])))
    print('All words accuracy: {}'.format(accuracy_score(data[options.to_lang], [v[0][0] if len(v) > 0 else 'dummy_word_123987' for v in data.pred])))
    data['pred_out'] = [v[0][0] if len(v) > 0 else 'dummy_word_123987' for v in data.pred]

    th_from = options.th_from
    th_step = options.th_step
    th_to = options.th_to

    max_all = -1.0
    th_all = None
    max_contained = -1.0
    th_contained = None
    min_sim = min([v for i in data.pred for w,v in i])
    for th in tqdm(list(np.arange(max(th_from, min_sim), th_to, th_step))):
        tmp_all = eval_with_th(data[options.from_lang].tolist(), data[options.to_lang].tolist(), data.pred.tolist(), th)
        if tmp_all > max_all:
            max_all = tmp_all
            th_all = th

        tmp_contained = eval_with_th(data[data.contains][options.from_lang].tolist(), data[data.contains][options.to_lang].tolist(), data[data.contains].pred.tolist(), th)
        if tmp_contained > max_contained:
            max_contained = tmp_contained
            th_contained = th

    print('Contained words F1 with threshold ({}): {}'.format(th_contained, max_contained))
    print('All words F1 with threshold ({}): {}'.format(th_all, max_all))

    if options.out is not None:
        data.ix[:, ['en', 'nl', 'pred_out']].to_csv(options.out, sep='\t')
