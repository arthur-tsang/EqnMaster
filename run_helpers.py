# This file provides some helpers for running stuff

import numpy as np

from misc import get_data

# Keep track of vocab as a string of certain characters
outvocab = '0123456789'
invocab = outvocab + ' +*'
# outdim = len(outvocab)
# vdim = len(invocab)
# Dictionary to look up index by character
outdico = {c:i for i,c in enumerate(outvocab)}
indico = {c:i for i,c in enumerate(invocab)}


def encode(nr_string, dico=indico, asNumpy = True):
    # using indico as default, because that's more general
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
    return decode(model.generate_answer(encode(in_string, indico)), outvocab)

def preprocess_data(train_file, asNumpy = True):
    """filename should be models/<whatever>.p
    Returns X and Y as lists (per data example) of lists (per character) of indices"""

    ## Get data
    raw_data = get_data(train_file)
    X, Y = labelize(raw_data, asNumpy = asNumpy)
    
    return X,Y
