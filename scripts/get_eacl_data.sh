#!/bin/bash

tmpf=`mktemp`

wget http://liir.cs.kuleuven.be/software/BLI_classifier.zip -O $tmpf
unzip $tmpf 'BLI_classifier/eacl_data/*'

rm $tmpf