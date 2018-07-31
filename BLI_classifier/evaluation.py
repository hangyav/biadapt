from __future__ import division

from collections import namedtuple

Result = namedtuple('Result', ['f1', 'precision', 'recall'])


def eval_translations(groundtruth, translations):
  groundtruth_set = set(groundtruth)
  true_positives = groundtruth_set & translations
  recall = (len(true_positives) / len(groundtruth_set) * 100)
  precision = (len(true_positives) / len(translations) * 100) \
    if translations else float('nan')
  f1 = 2 * recall * precision / (recall + precision) \
    if (recall + precision) else float('nan')
  result = Result(f1, precision, recall)
  return result


def extract_translations(training_lexicon, source_words, predicted_targets):
  def correct_training_translation(w_s, w_t):
    if w_s in training_lexicon:
      return w_t in training_lexicon[w_s]
    else:
      return False

  translation_pairs = set()
  top_1_translation_pairs = set()
  for w_s, w_preds in zip(source_words, predicted_targets):
    if not w_preds:
      continue
    added_top1 = False
    for w_pred, _ in w_preds:
      if not correct_training_translation(w_s, w_pred):
        translation_pairs.add((w_s, w_pred))
        if not added_top1:
          top_1_translation_pairs.add((w_s, w_pred))
          added_top1 = True
  return top_1_translation_pairs, translation_pairs
