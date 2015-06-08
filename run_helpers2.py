# This file provides some helpers for running stuff

# This file is a repeat of run_helpers.py, with the one exception that it allows us to output minus signs. This will become the default after our poster presentation.

import numpy as np

from misc import get_data

# Keep track of vocab as a string of certain characters
outvocab = '0123456789-'
invocab = outvocab + ' +*'
invocab2 = invocab + '='
# outdim = len(outvocab)
# vdim = len(invocab)
# Dictionary to look up index by character
outdico = {c:i for i,c in enumerate(outvocab)}
indico = {c:i for i,c in enumerate(invocab)}
indico2 = {c:i for i,c in enumerate(invocab2)}


def encode(nr_string, dico=indico2, asNumpy = True):
    # using indico2 as default, because that's more general
    list_encoding = [dico[c] for c in nr_string]
    return np.array(list_encoding) if asNumpy else list_encoding

def decode(labels, vocab=outvocab):
    # make sure to use invocab if you want to make input sequences look human-readable
    return ''.join(vocab[label] for label in labels if label<len(vocab))

def labelize(xy_data, asNumpy = True):
    # Go from tuples of strings to X and Y as lists of int labels
    X = [encode(x, indico, asNumpy = asNumpy) for x,y in xy_data]
    Y = [encode(y, outdico, asNumpy = asNumpy) for x,y in xy_data]

    return (X,Y)

def model_solve(model, in_string):
    return decode(model.generate_answer(encode(in_string, indico2)), outvocab)

def model_solve_discr(model, in_string):
    return model.generate_answer(encode(in_string, indico2))

def preprocess_data(train_file, asNumpy = True):
    """filename should be models/<whatever>.p
    Returns X and Y as lists (per data example) of lists (per character) of indices"""

    ## Get data
    raw_data = get_data(train_file)
    X, Y = labelize(raw_data, asNumpy = asNumpy)
    
    return X,Y

def extract_discr_data(train_file, asNumpy = True):
    raw_data = get_data(train_file)
    Y = [y for (_, y) in raw_data]
    X = [encode(x, indico2, asNumpy = asNumpy) for x,y in raw_data]
    return X, Y
