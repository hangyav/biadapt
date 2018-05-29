from __future__ import division
from __future__ import print_function


import numpy as np
from io import BytesIO
from tensorflow.python.lib.io import file_io


NUM_LABELS = 3
IMAGE_SHAPE = [81]

def get_data(name, dir):
    if name not in ['train', 'test', 'unlabeled']:
        raise ValueError('{} is not in the dataset!'.format(name))

    data = np.load(BytesIO(file_io.read_file_to_string('{}/{}.npz'.format(dir, name))))
    seqs = data['data']
    labels = None

    if name != 'unlabeled':
        labels = data['label']

    return seqs, labels