from nltk.tokenize import TweetTokenizer

import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)
from nltk.stem import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd


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
