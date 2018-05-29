from __future__ import division
from __future__ import print_function

from .data_dirs import numpy_input_path
import numpy as np


NUM_LABELS = 3
IMAGE_SHAPE = [81]

def get_data(name):
    if name not in ['train', 'test', 'unlabeled']:
        raise ValueError('{} is not in the dataset!'.format(name))

    data = np.load('{}/{}.npz'.format(numpy_input_path, name))
    seqs = data['data']
    labels = None

    if name != 'unlabeled':
        labels = data['label']

    return seqs, labels