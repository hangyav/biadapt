import logging
from optparse import OptionParser
import numpy as np
import utils as u
import pandas as pd
import pickle


def get_options():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data", type=str, action='callback', callback=u.str_list_callback)
    parser.add_option("--do", "--dictionary_output", dest="dictionary_output")
    parser.add_option("-o", "--output", dest="output", type=str, action='callback', callback=u.str_list_callback)
    parser.add_option("-w", "--window", dest="window", default=5)

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('preprocess_data_for_semisup.py')
    options = get_options()

    dfs = list()

    for i, path in enumerate(options.data):
        logger.info('Loading and preprocessing data {}'.format(path))
        data = u.preprocess_data(path)
        data['idx'] = i
        dfs.append(data)

    df = pd.concat(dfs)

    logger.info('Converting to sequence...')
    seqs, tokenizer, nb_words, seq_len = u.to_sequence(df.pp, window=options.window)
    labels, encoder = u.labelencode(df.label)

    df['seqs'] = seqs.tolist()
    df['onehot'] = labels.tolist()

    for i, path in zip(range(df.idx.max()+1), options.output):
        logger.info('Saving data {}'.format(path))
        seqs = np.array([v for v in df[df.idx == i].seqs])
        labels = np.array([v for v in df[df.idx == i].onehot])

        np.savez(path, data=seqs, label=labels)

    logger.info('Saving tokenizers {}'.format(options.dictionary_output))
    with open(options.dictionary_output, 'wb') as fout:
        pickle.dump([tokenizer, encoder, nb_words, seq_len], fout, protocol=2)