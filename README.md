# Methods for Domain Adaptation of Bilingual Tasks

This repository contains implementation of the work *[Two Methods for Domain Adaptation of Bilingual Tasks: Delightfully Simple and Broadly Applicable](http://aclweb.org/anthology/P18-1075)*.

## Cite

```
@InProceedings{P18-1075,
  author = 	"Hangya, Viktor
  		and Braune, Fabienne
		and Fraser, Alexander
		and Sch{\"u}tze, Hinrich",
  title = 	"Two Methods for Domain Adaptation of Bilingual Tasks: Delightfully Simple and Broadly Applicable",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"810--820",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1075"
}

```

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

* For the semi-supervised system for sentiment we modified the original implementation of [(Haeusser et al. 2017)](https://github.com/haeusser/learning_by_association)
* We added the implementation of [(Kim (2014)’s CNN-non-static)](https://github.com/yoonkim/CNN_sentence)
* An example script demonstrating the use of the system: *scripts/run_semisup_sentiment.sh*

## Bilingual Lexicon Induction


### Cosine similarity

* *scripts/bll_with_threshold.py*: also use for fine tuning of the threshold on the developement set (use *-h* to get input parameters)

### Classifier

TODO

### Semi-supervised

TODO
