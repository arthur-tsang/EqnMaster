#!/usr/bin/env python

import numpy as np

from enc_dec import EncDec
from misc import get_data

# Keep track of vocab as a string of certain characters
outvocab = '0123456789'
invocab = outvocab + ' +'
outdim = len(outvocab)
vdim = len(invocab)
# Dictionary to look up index by character
outdico = {c:i for i,c in enumerate(outvocab)}
indico = {c:i for i,c in enumerate(invocab)}


def encode(nr_string, dico):
    return np.array([dico[c] for c in nr_string])

def decode(labels, vocab):
    return ''.join(vocab[label] for label in labels if label<len(vocab))

def labelize(xy_data):
    # Go from tuples of strings to X and Y as lists of int labels
    X = [encode(x, indico) for x,y in xy_data]
    Y = [encode(y, outdico) for x,y in xy_data]

    return (X,Y)

def ed_solve(ed, in_string):
    return decode(ed.generate_answer(encode(in_string, indico)), outvocab)

if __name__ == '__main__':
    ## Filename to save ED
    model_filename = 'ed_full.p'

    ## Data
    train_data = get_data('data/3dig_train.p')
    dev_data = get_data('data/3dig_dev.p')

    X_train, Y_train = labelize(train_data)
    X_dev, Y_dev = labelize(dev_data)

    ## Hyperparameters
    hdim = 10
    wdim = 10
    batch_size = 5
    n_epochs = 200
    
    ## EncDec model train
    ed = EncDec(vdim, hdim, wdim, outdim)
    ed.grad_check(X_train[:10], Y_train[:10])
    # ed.load_model(model_filename) # if retraining
    ed.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=X_dev, Y_dev=Y_dev, verbose=True)
    ed.save_model(model_filename)

    ## EncDec model test
    # toy_problems = ['5+15','17+98','7+7','3+7']
    # ed.load_model(model_filename)

    # for toy in toy_problems:
    #     print toy,'=',ed_solve(ed, toy)
