from __future__ import print_function

import codecs
import numpy as np
import xml.etree.ElementTree as ET


def load_embeddings(filename, encoding='utf-8', binary=True, max=None, mwu_delim=None):
  with open(filename, 'rb') as f:
    meta_info = f.readline().strip().split()
    vocab_size, emb_dim = int(meta_info[0]), int(meta_info[1])
    vocab = []
    vocab_size = min(max, vocab_size) if max else vocab_size
    embeddings = np.zeros(shape=(vocab_size, emb_dim), dtype=np.float32)
    for i in xrange(vocab_size):
      if max and max == i:
        break
      if (i + 1) % 10 ** 5 == 0:
        print('Loaded %i of %i embeddings' % (i + 1, vocab_size))
      if binary:
        word, embedding = _read_wordembedding_bin(f, emb_dim, np.float32, encoding)
      else:
        word, embedding = _read_wordembedding_txt(f, emb_dim, np.float32, encoding)
      if mwu_delim:
        word = word.replace(mwu_delim, ' ')
      vocab.append(word)
      embeddings[i] = embedding

  return vocab, embeddings


def _read_wordembedding_txt(f, emb_dim, dtype, encoding):
  l = f.readline().strip()
  l_split = l.split()
  word = l_split[0].decode(encoding)
  embedding = map(float, l_split[1:])
  embedding = np.array(embedding, dtype)
  return word, embedding


def _read_wordembedding_bin(f, emb_dim, dtype, encoding):
  word = []
  char = f.read(1)
  if char == b'\n':
    char = ''
  while char != b' ':
    word.append(char)
    char = f.read(1)
  word = b''.join(word).decode(encoding)
  itemsize = np.dtype(dtype).itemsize
  embedding_size = itemsize * emb_dim
  emb_str = f.read(embedding_size)
  embedding = np.fromstring(emb_str, dtype)
  return word, embedding


def save_embeddings(filename, embeddings, words, encoding='utf-8'):
  vocab_size, emb_dim = embeddings.shape
  if vocab_size != len(words):
    raise ValueError("Embeddings and words don't match")
  with open(filename, 'wb') as f:
    f.write('%i %i\n' % (vocab_size, emb_dim))
    for i, word in enumerate(words):
      f.write(word.encode(encoding=encoding))
      f.write(b' ')
      wordemb = embeddings[i]
      f.write(wordemb.tostring())


def load_bilingual_embs(embs, suffixes, encoding='utf-8', binary=True):
  words, E = load_embeddings(embs)
  words_s, E_s = [], []
  words_t, E_t = [], []
  for i, word in enumerate(words):
    if word[-2:] == suffixes[0]:
      words_s.append(word[:-3])
      E_s.append(E[i])
    else:
      words_t.append(word[:-3])
      E_t.append(E[i])
  E_s = np.array(E_s)
  E_t = np.array(E_t)
  print(E_s.shape)
  return words_s, words_t, E_s, E_t


def load_lexicon(lexicon_f):
  lexicon = []
  with codecs.open(lexicon_f, 'rb', 'utf-8') as f:
    for l in f:
      wordpair = l.strip().split('\t')
      lexicon.append(tuple(wordpair))
  return lexicon


def save_lexicon(lexicon, lexicon_f):
  with codecs.open(lexicon_f, 'wb', 'utf-8') as f:
    for w_s, w_t in lexicon:
      f.write('%s\t%s\n' % (w_s, w_t))


def embeddings_and_vocab(embedding_file, mwu_delim=None, binary=True):
  print('Loading embeddings')
  words, embeddings = load_embeddings(embedding_file, binary=binary, mwu_delim=mwu_delim)
  words_source = []
  words_target = []
  embeddings_source = []
  embeddings_target = []
  for i, w in enumerate(words):
    if '</s>' in w:
      continue
    if len(w) == 3:
      continue
    en_suffix = ' en' if mwu_delim == '_' else '_en'
    if w[-3:] == en_suffix:
      words_source.append(w[:-3])
      embeddings_source.append(embeddings[i])
    else:
      words_target.append(w[:-3])
      embeddings_target.append(embeddings[i])
  vocabulary = words_source + words_target
  embeddings_source = np.array(embeddings_source)
  embeddings_target = np.array(embeddings_target)
  print('Number of source words:', len(words_source))
  print('Number of target words:', len(words_target))
  return words_source, words_target, vocabulary, embeddings_source, embeddings_target, embeddings


def char_vocab(words_source, words_target, word_delimiters=True):
  chars_source = set(c for w in words_source for c in w)
  chars_target = set(c for w in words_target for c in w)
  chars_source = sorted(chars_source)
  chars_target = sorted(chars_target)
  special_chars = ['<pad>']
  if word_delimiters:
    special_chars.append('<bow>')
    special_chars.append('<eow>')
  return special_chars + chars_source, special_chars + chars_target


class IATE(object):

  def __init__(self, tbx_file):
    tree = ET.parse(tbx_file)
    self._root = tree.getroot()

  def _term_entries(self):
    return self._root.iter('termEntry')

  def term_pairs(self, source_lang, target_lang):
    xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
    lang_attrib = xml_namespace + 'lang'
    for term_entry in self._term_entries():
      source_terms = []
      target_terms = []
      for lang_set in term_entry.iter("langSet"):
        if lang_set.attrib[lang_attrib] == source_lang:
          for term in lang_set.iter('term'):
            source_terms.append(term.text)
        if lang_set.attrib[lang_attrib] == target_lang:
          for term in lang_set.iter('term'):
            target_terms.append(term.text)
      for source_term in source_terms:
        for target_term in target_terms:
          yield source_term, target_term

  def terms(self, lang):
    xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
    lang_attrib = xml_namespace + 'lang'
    for term_entry in self._term_entries():
      for langSet in term_entry.iter("langSet"):
        if langSet.attrib[lang_attrib] == lang:
          for term in langSet.iter('term'):
            yield term.text



if __name__ == '__main__':
  embeddings_fn = '/Users/geert/code/ble_classifier/data/embeddings/ennl.bwesg.mwu.dim=50.window=100.bin'
  embeddings_fn = '/Users/geert/code/ble_classifier/data/embeddings/ennl.mono.mwu.dim=50.bin'

  words, embeddings = load_embeddings(embeddings_fn, binary=True)
  print(words)
  #save_embeddings(embeddings=embeddings[1:], words=words[1:], filename=embeddings_fn)
