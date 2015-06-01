#!/usr/bin/env python

import numpy as np

from enc_dec import EncDec
from misc import get_data
# from visualize_vecs import svd_visualize, tsne_visualize, multi_tsne, pca_visualize

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
    model_filename = 'models/ed_full.p'

    ## Data
    train_data = get_data('data/3dig_train.p')
    dev_data = get_data('data/3dig_dev.p')

    X_train, Y_train = labelize(train_data)
    X_dev, Y_dev = labelize(dev_data)

    ## Hyperparameters
    hdim = 50
    wdim = 50
    batch_size = 1000
    n_epochs = 5000

    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    
    ## EncDec model train
    ed = EncDec(vdim, hdim, wdim, outdim, alpha=0.001, rho = 0.0000)
    #ed.grad_check(X_train[:10], Y_train[:10])
    ed.load_model(model_filename) # if retraining
    ed.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True)
    ed.save_model(model_filename)

    ## EncDec model test
    toy_problems = [decode(x, invocab) for x in X_train]
    # toy_problems = ['5+15','17+98','7+7','3+7']
    
    L = ed.encoder.params['L']
    #svd_visualize(np.transpose(L), invocab)
    pca_visualize(np.transpose(L), invocab)
    #multi_tsne(np.transpose(L), invocab)

    # for toy in toy_problems:
    #     print toy,'=',ed_solve(ed, toy)

# Note that we can get the following kind of error during training:
"""
/home/arthur/Documents/cs224d/EqnMaster/enc.py:76: RuntimeWarning: divide by zero encountered in log
  cost += -np.log(yhat[ys[t]])
/home/arthur/Documents/cs224d/EqnMaster/dec.py:77: RuntimeWarning: divide by zero encountered in log
  cost += -np.log(yhat[ys[t]])
/home/arthur/Documents/cs224d/EqnMaster/nn/math.py:7: RuntimeWarning: invalid value encountered in subtract
  xt = exp(x - max(x))
"""
# TODO: how should we account for this?
# Remark: theano has something called local_log_softmax (stabilitzes softmax so it doesn't have us take the log of 0)
# :)
