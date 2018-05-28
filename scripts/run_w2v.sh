#!/bin/bash

if [ $# -ne  2 ]; then
	echo "$0 <input> <output>"
	exit 1
fi

tmpf=`mktemp`

./word2vec -train $1 -output $tmpf -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 15 -binary 0
echo -e '\nconverting...'
iconv -f 'utf-8' -t 'utf-8' -c $tmpf > $2

rm $tmpf
