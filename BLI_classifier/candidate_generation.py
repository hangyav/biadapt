from __future__ import print_function

import random

import numpy as np
from tqdm import tqdm


class CandidateGenerator(object):
  def __init__(self, words_s, words_t, embeddings_s, embeddings_t, num_candidates, edit_distance, weight=0.5, njobs=4):
    self.num_candidates = num_candidates
    # self._words_s = words_s
    # self._words_t = words_t
    # self._edit_distance = edit_distance
    # self._weight = weight
    self._init_candidates(words_s, words_t, embeddings_s, embeddings_t, njobs, edit_distance, weight)

  def _init_candidates(self, words_s, words_t, embeddings_s, embeddings_t, njobs, edit_distance, weight):
    with tqdm(total=4) as pbar:
      self._candidates_source = {}
      self._candidates_target = {}
      num_candidates = self.num_candidates
      num_edit_distance = int(num_candidates * weight)
      print('  Generating edit distance candidates.')
      edit_distance_matrix_st = edit_distance.get_edit_distance_matrix(njobs=njobs)
      min_edit_indices_st = edit_distance_matrix_st.argsort(axis=1)[:, :num_edit_distance]
      edit_distance_matrix_ts = edit_distance_matrix_st.transpose()
      min_edit_indices_ts = edit_distance_matrix_ts.argsort(axis=1)[:, :num_edit_distance]
      for i, w_s in enumerate(tqdm(words_s)):
        indices = min_edit_indices_st[i]
        self._candidates_source[w_s] = set(words_t[j] for j in indices)
      pbar.update(1)
      for i, w_t in enumerate(tqdm(words_t)):
        indices = min_edit_indices_ts[i]
        self._candidates_target[w_t] = set(words_s[j] for j in indices)
      pbar.update(1)
      del edit_distance_matrix_st
      del edit_distance_matrix_ts
      del min_edit_indices_st
      del min_edit_indices_ts
      print('  Generating embedding candidates.')
      num_embeddings = num_candidates - num_edit_distance
      embeddings_s_n = embeddings_s / np.linalg.norm(embeddings_s, axis=1, keepdims=True)
      embeddings_t_n = embeddings_t / np.linalg.norm(embeddings_t, axis=1, keepdims=True)
      embs_scores = np.dot(embeddings_s_n, embeddings_t_n.T)
      embs_scores_ts = np.transpose(embs_scores)
      embs_scores_sorted = np.argsort(embs_scores, axis=1)  # num_source x num_target
      for i, w_s in enumerate(tqdm(words_s)):
        j = 0
        for _ in xrange(num_embeddings):
          w_t = words_t[embs_scores_sorted[i][-j - 1]]
          while w_t in self._candidates_source[w_s]:
            j += 1
            w_t = words_t[embs_scores_sorted[i][-j-1]]
          self._candidates_source[w_s].add(w_t)
      pbar.update(1)

      embs_scores_ts_sorted = np.argsort(embs_scores_ts, axis=1)  # num_target x num_source
      for i, w_t in enumerate(tqdm(words_t)):
        j = 0
        for _ in xrange(num_embeddings):
          w_s = words_s[embs_scores_ts_sorted[i][-j-1]]
          while w_s in self._candidates_target[w_t]:
            j += 1
            w_s = words_s[embs_scores_ts_sorted[i][-j-1]]
          self._candidates_target[w_t].add(w_s)
      pbar.update(1)

  def candidates(self, w, source2target, w_true=None):
    candidates = self._candidates_source if source2target else self._candidates_target
    candidates = candidates[w] if w in candidates else []
    candidates = [w for w in candidates if w != w_true]
    return candidates

  def translation_in_candidates(self, w, w_true, source2target):
    candidates = self._candidates_source if source2target else self._candidates_target
    candidates = candidates[w]
    return w_true in candidates


class RandomCandidateGenerator(object):
  def __init__(self, words_s, words_t, num_candidates):
    self.num_candidates = num_candidates
    self._words_s = list(words_s)
    self._words_t = list(words_t)

  def candidates(self, w, source2target, w_true=None):
    words = self._words_t if source2target else self._words_s
    candidates = set()
    while len(candidates) != self.num_candidates:
      word = random.choice(words)
      if word != w_true:
        candidates.add(word)
    return list(candidates)

class MixedCandidateGenerator(object):
  def __init__(self, similarity_generator, random_generator):
    self.similarity_generator = similarity_generator
    self.random_generator = random_generator

  def candidates(self, w, source2target, w_true=None):
    cand1 = self.similarity_generator.candidates(w, source2target, w_true)
    cand2 = self.random_generator.candidates(w, source2target, w_true)

    return cand1[:int(len(cand1)/2)] + cand2[:int(len(cand2)/2)]