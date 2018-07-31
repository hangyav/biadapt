# Biligual Lexicon Induction by Combining Word-level and Character-level Information
In this repository you find code for the EACL [paper](https://drive.google.com/file/d/0Byj5g4SlS5SDSXRoeWo4YjRuOTg/view):


```
@InProceedings{Heyman:2017,
  author    = {Heyman, Geert  and Vuli\'{c}, Ivan and Moens, Marie-Francine},
  title     = {Bilingual Lexicon Induction by Learning to Combine Word-Level and Character-Level Representations},
  booktitle = {Proceedings of EACL 2017},
  year      = {2017},
  pages     = {1084--1094},
}
```

## Installation
The code was written in python 2.7. You can install the dependencies with:
```pip install -r requirements.txt```

## Creating a classifier
A classifier is created as follows:
```python
with tf.Session() as session:
  classifier = Classifier(session, vocab_S, vocab_T, Features(features), negative_sampling, candidate_generation)
```

* `vocab_S`, `vocab_T` are vocabularies for the source/target language respectively.
* `features` is a list of the features/representations the classifier will use. These can be both manually defined 
(e.g., edit distance) or learned from the training lexicon (e.g., the output of a character-level LSTM).
* `negative_sampling` is the strategy for sampling negative/noise examples during training.
* `candidate_generation` is the strategy for sampling candidate translation pairs during inference. 

### Features/representations
You can add *manually* defined features, pre-trained features or features that are learned.

Manually defined features are grouped in the `feature` module. This is how you create a normalized edit distance feature:
```python
edit_distance = features.EditDistance(vocab_S, vocab_T, 'edit_distance.npy')
normalized_edit_distance = features.NormalizedEditDistance(edit_distance)
```
* `edit_distance.npy` is a filename which will be used to save/load a matrix with all edit distances so that in
subsequent runs the calculation of all edit distances does not need to be repeated.

Trained features are grouped in the `wordencoding` module. The character-level LSTM encoder that is used in the paper can be instantiated as follows:
```python
charlevel_features_encoder = wordencoding.BilingualRNNEncoding(char_vocab_S, char_vocab_T, num_cells)
charlevel_features = char_features_encoder(maximum_length)
```
* `num_cells` is a list or int that specifies the dimensions of the LSTM. For example, `num_cells = [128,128]` creates a two-layer LSTM with 128 cells in each layer.
* `maximum_length` should be set to the character-length of the longest word in the vocabularies.
To use pre-trained word embeddings in your classifier you can do the following:
```python
wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_S, embeddings=Embs_S, scope='source')
word_embs_S = wordlevel_features_encoder()
wordlevel_features_encoder = wordencoding.WordLevelEncoding(vocab_T, embeddings=Embs_T, scope='target')
word_embs_T = wordlevel_features_encoder()
```
* `Embs_S`, `Embs_T` are word embedding matrices for the source/target language.

After having defined all the features they are wrapped inside a Feature instance:
```python
feature.Features([normalized_edit_distance, charlevel_features, word_embs_S, word_embs_T])
```

### Choosing the strategy to sample negative/noise examples
In the paper we randomly sample negative/noise samples. For this use the RandomCandidateGenerator class in the `candidate_generation` module.
```python
negative_sampling = candidate_generation.RandomCandidateGenerator(vocab_S, vocab_T, num_neg_samples)
```

### Choosing candidate generation for inference
To use the candidate generation that was used in the paper use the CandidateGenerator class in the `candidate_generation` module.
```python
candidate_generation = candidate_generation.CandidateGenerator(vocab_S, vocab_T, multi_Embs_S, multi_Embs_T, num_candidates, edit_distance)
```
* `multi_Embs_S`, `multi_Embs_T` are word embedding matrices. It is important that the embeddings of source and target language lie in the same space. In the paper we use BWESG embeddings for candidate generation.


## Training the classifier
First create an instance of the `Trainer` class of the `trainer` module.

```python
training_data_feeder = BilexDataFeeder(training_data, batch_size, shuffle=True)
trainer = Trainer(classifier, num_epochs, training_data_feeder)
```
* `BilexDataFeeder` shuffles the training data and divides it into batches.

Before you start training there is the option of adding some extra evaluation/logging functionality:

```python
trainer.add_command(EpochLossLogger(classifier, log_dir))
trainer.add_command(BasicStatsLogger(classifier, training_data_feeder, num_epochs, period))
trainer.add_command(Evaluation(classifier, test_data_feeder, training_lexicon, num_epochs, log_dir))
```

`trainer.train()` starts training.

## Predicting translations

Once training is done you can predict translations with `classifier.predict()`

```python
# predict from source to target
translations_T = classifier.predict(source_words, source2target=True, threshold=0.5)
# predict from target to source
translations_S = classifier.predict(target_words, source2target=False, threshold=0.5)
```


