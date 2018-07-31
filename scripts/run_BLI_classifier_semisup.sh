#!/bin/bash

#PYTHON2=python2.7
#PYTHON3=python3.5
PYTHON2=/mounts/Users/student/hangyav/.anaconda/envs/bwe_adapt2/bin/python
PYTHON3=/mounts/Users/student/hangyav/.anaconda/envs/bwe_adapt/bin/python
THREADS=10
GPUS=0

# to merge separate source/target embedding files use:  BLI_classifier/convert_mwe2bwe.py
BWE_model='BLI_classifier/eacl_data/ennl.bwesg.dim=50.window=100.bin'
# change this to the produced adapted BWE model
adapted_BWE_model=$BWE_model
MWE_model='BLI_classifier/eacl_data/ennl.mono.dim=50.bin' # can be BWE as well
# source (En) and target (Nl) language (mapped) adapted embeddings (only needed with ruc=1)
SOURCE_BWE=/mounts/work/hangyav/project_tmp/BLI/vec/heyman_articles_nl__europarl-v7.nl-en.nl.clean.vec
TARGET_BWE=/mounts/work/hangyav/project_tmp/BLI/vec/mapped/heyman_articles_en__europarl-v7.nl-en.en.clean__mapped_to_nl__lex.filtered.train80-20.vec
train_lexicon=BLI_classifier/eacl_data/lex.filtered.train80-20.txt
test_lexicon=BLI_classifier/eacl_data/lex.filtered.test80-20.txt

ww=1.0
vw=0.5
lw=1.0
# set this to 0 in order to use BNC lexicon as unlabeled or 1 in order to generate reverse medical lexicon
ruc=0

mkdir -p tmp/BLI_classifier/semisup


echo -e 'Generating candidates...'
$PYTHON2 BLI_classifier/candidate_generator.py --num_candidates 10 --bwesg_embedding_file $BWE_model --output tmp/semisup_candidates.pkl --editdistance_file tmp/semisup_edit_distance.npy --threads $THREADS

if [ $ruc -eq 1 ]; then
    echo -e 'Generating reverse lexicon...'
    tmpf=`mktemp`
    echo -e 'en\tnl' > $tmpf
    cat $train_lexicon >> $tmpf
    $PYTHON3 scripts/reverse_lexicon_generator.py --num_candidates 3 --from_vectors $SOURCE_BWE --to_vectors $TARGET_BWE --from_lang en --to_lang nl --output tmp/semisup_lexicon.txt --input $tmpf
    rm $tmpf
else
    echo -e 'Using BNC as unlabeled lexicon...'
    cut -f 3,6 data/BNC/bnc.tsv > tmp/semisup_lexicon.txt
fi

echo -e 'Running experiment...'
CUDA_VISIBLE_DEVICES=$GPUS $PYTHON2 BLI_classifier/semisup_model.py --embedding_file $adapted_BWE_model --out_dir tmp/BLI_classifier/semisup \
    --training_data $train_lexicon --unlabeled_data tmp/semisup_lexicon.txt --test_data $test_lexicon --test_candidates_file tmp/semisup_candidates.pkl \
    --unlabeled_candidates_file tmp/semisup_candidates.pkl --walker_weight $ww --visit_weight $vw --logit_weight $lw --add_negative_unlabeled 1 --reversed_unlabeled_candidates $ruc