#!/bin/bash

PYTHON=python

# Format of datasets (header included): id,label,text
labeled_data=
unlabeled_data=
dev_data=

# Source and target language bilingual word embeddings
w2v1=
w2v2=


mkdir -p tmp

echo 'Preprocessing...'
$PYTHON scripts/preprocess_data_for_semisup_sentiment.py -d ${labeled_data},${unlabeled_data},${dev_data} -o tmp/train,tmp/unlabeled,tmp/test --do tmp/tokenizer.pkl
echo "numpy_input_path = 'tmp'" > target-ignorant_with_semisup/semisup/tools/data_dirs.py

echo 'Running semisup...'
PYTHONPATH=target-ignorant_with_semisup/ $PYTHON  target-ignorant_with_semisup/semisup/train.py --dataset numpy_data --architecture cnn_sentiment_model --logdir tmp/ --sup_per_class -1 --sup_seed 0 --sup_per_batch 10 --unsup_batch_size 100 --emb_size 128 --visit_weight 0.5 --walker_weight 1.0 --logit_weight 1.0 --dictionaries tmp/tokenizer.pkl --w2v ${w2v1},${w2v2} --static_word_embeddings 1 --max_steps 5000 --log_every_n_steps 100 --optimizer None --embedding_dropout 0.2 --learning_rate 0.0001 --decay_factor 0.7 --decay_steps 5000 --shuffle_input True --num_cpus 20 --walker_weight_envelope None
