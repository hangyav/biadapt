import logging
from nltk.tokenize import TweetTokenizer
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)
from nltk.stem import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

logger = logging.getLogger(__name__)

def int_list_callback(option, opt, value, parser):
    """
    Helper function for OptionParser to parse list of ints.
    :param option:
    :param opt:
    :param value:
    :param parser:
    :return:
    """
    data = value.split(',')
    setattr(parser.values, option.dest, [int(i) for i in data])


def float_list_callback(option, opt, value, parser):
    """
    Helper function for OptionParser to parse list of floats.
    :param option:
    :param opt:
    :param value:
    :param parser:
    :return:
    """
    data = value.split(',')
    setattr(parser.values, option.dest, [float(i) for i in data])


def str_list_callback(option, opt, value, parser):
    """
    Helper function for OptionParser to parse list of strings.
    :param option:
    :param opt:
    :param value:
    :param parser:
    :return:
    """
    data = value.split(',')
    setattr(parser.values, option.dest, [str(i) for i in data])


def to_iob(text, target, label, tokenizer=TweetTokenizer()):
    text = tokenizer.tokenize(text)
    target = target.lower().split()
    found = False

    res = list()
    i = 0
    while i < len(text):
        if found or target[0] != text[i]:
            res.append((text[i], 'o'))
        else:
            if len(target) == 1:
                res.append((text[i], 'b-{}'.format(label)))
                found = True
            else:
                good = True
                for j in range(1, len(target)):
                    if i+j == len(text):
                        good = False
                        break
                    if target[j] != text[i+j]:
                        good = False
                        break
                if good:
                    res.append((text[i], 'b-{}'.format(label)))
                    for j in range(1, len(target)):
                        res.append((text[i+1], 'i-{}'.format(label)))
                        i += 1
                    found = True

        i += 1

    if found:
        return res

    res = list()
    i = 0
    while i < len(text):
        if found or  (target[0] not in text[i]):
            res.append((text[i], 'o'))
        else:
            for t in target:
                if t in text[i]:
                    res.append((text[i], '{}-{}'.format(('b' if t == target[0] else 'i'), label)))
                    i += 1
                    found = True
                    if i == len(text):
                        break
                else:
                    break
            i -= 1

        i += 1

    if found:
        return res

    res.append(('</s>', 'o'))
    for t in target:
        res.append((t, '{}-{}'.format(('b' if t == target[0] else 'i'), label)))

    return res


def preprocess_tweets(text, lower=True, cleaner=p.clean, tokenizer=TweetTokenizer().tokenize,
        stopwords=None):
    if lower:
        text = text.lower()
    if cleaner is not None:
        text = cleaner(text)
    if tokenizer is not None:
        text = tokenizer(text)
    if stopwords is not None:
        if tokenizer is None:
            raise ValueError('Need nokenizer')
        text = [w for w in text if w.lower() not in stopwords]

    if tokenizer is not None:
        text = ' '.join(text)
    return text

def get_topn(words, sw2v, tw2v, topn=1, stopwords=None):
    res = list()
    for w in words:
        if stopwords is not None and w in stopwords:
            continue
        if w in sw2v:
            d = {'word': w}
            for i, item in enumerate(tw2v.most_similar([sw2v[w]], topn=topn)):
                d['neighbor{}'.format(i)] = item[0]
                d['similarity{}'.format(i)] = item[1]
            res.append(d)
    return pd.DataFrame(res)


def preprocess_data(data, col='text', new_col='pp'):
    df = pd.read_csv(data, index_col=0)

    # p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)

    df[new_col] = df[col].str.lower()
    df[new_col] = df[col].apply(p.clean)

    return df


def to_sequence(texts, window=5, maxlen=None):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    nb_words = len(tokenizer.word_index.items()) + 1

    if maxlen is None:
        maxlen = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])

    logger.info('Maximum sentence length: {}'.format(maxlen))
    logger.info('Padded sentence length: {}'.format(maxlen + 2 * (window - 1)))
    logger.info('Number of words: {}'.format(nb_words))

    maxlen += window - 1

    seqs = tokenizer.texts_to_sequences(texts)
    seqs = sequence.pad_sequences(seqs, padding='post', maxlen=maxlen)
    seqs = sequence.pad_sequences(seqs, padding='pre', maxlen=maxlen + window - 1)

    return seqs, tokenizer, nb_words, maxlen + (window - 1)


def onehotencode(labels):
    encoder = LabelBinarizer()
    res = encoder.fit_transform(labels)
    return res, encoder


def labelencode(labels):
    encoder = LabelEncoder()
    res = encoder.fit_transform(labels)
    return res, encoder
