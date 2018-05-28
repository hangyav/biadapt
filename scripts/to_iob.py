#!/usr/bin/env python

import logging
from optparse import OptionParser

import pandas as pd
import numpy as np
import utils as u


def get_options():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data")
    parser.add_option("-o", "--output", dest="output")

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    options = get_options()

    data = pd.read_csv(options.data)
    
    with open(options.output, 'w') as fout:
        for i,r in data.dropna().iterrows():
            for item in u.to_iob(r.cleaned_text, r.target, r.label):
                fout.write('{}\n'.format(' '.join(item)))
            fout.write('\n')
