#!/bin/bash

PYTHON=python
THREADS=10
GPUS=0

#
BWE_model='BLI_classifier/eacl_data/ennl.bwesg.dim=50.window=100.bin'
MWE_model='BLI_classifier/eacl_data/ennl.mono.dim=50.bin' # can be BWE as well
train_lexicon=BLI_classifier/eacl_data/lex.filtered.train80-20.txt
test_lexicon=BLI_classifier/eacl_data/lex.filtered.test80-20.txt

mkdir -p tmp/BLI_classifier

echo -e 'Generating candidates...'
$PYTHON BLI_classifier/candidate_generator.py --num_candidates 10 --bwesg_embedding_file $BWE_model --output tmp/candidates.pkl --editdistance_file tmp/edit_distance.npy --threads $THREADS

echo -e 'Running experiment classifier...'

CUDA_VISIBLE_DEVICES=$GPUS $PYTHON BLI_classifier/model.py --embedding_file $MWE_model --out_dir tmp/BLI_classifier --training_data $train_lexicon --test_data $test_lexicon --candidates_file tmp/candidates.pkl