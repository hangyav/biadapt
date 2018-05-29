# Methods for Domain Adaptation of Bilingual Tasks

TODO description

##Cite

TODO

## Requirements

* Python 3.5
* dependencies in requirements.txt

```sh
	pip install -r requirements.txt
```

## Cross Lingual Sentiment Classification

### Target-ignorant

* Follow the procedure at section _Semi-supervised_ below
* Set visit and walker weights to 0.0

### Target-aware system

* As target-aware system we used the method of [(Zhang et al., 2016)](https://github.com/SUTDNLP/NNTargetedSentiment)
* To convert data to iob format use the script: *scripts/to_iob.py*

### Semi-supervised

* For the semi-supervised system for sentiment we modified the original implementation of [(Haeusser et al. (2017))](https://github.com/haeusser/learning_by_association)
* We added the implementation of [(Kim (2014)’s CNN-non-static)](https://github.com/yoonkim/CNN_sentence)
* An example script demonstrating the use of the system: *scripts/run_semisup_sentiment.sh*

## Bilingual Lexicon Induction


### Cosine similarity

TODO

### Classifier

TODO

### Semi-supervised

TODO
