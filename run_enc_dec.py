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
    return ''.join(vocab[label] if label<len(vocab) else '</s>' for label in labels)

def labelize(xy_data):
    # Go from tuples of strings to X and Y as lists of int labels
    X = [encode(x, indico) for x,y in xy_data]
    Y = [encode(y, outdico) for x,y in xy_data]

    return (X,Y)

if __name__ == '__main__':
    train_data = get_data('data/train.txt')
    dev_data = get_data('data/dev.txt')

    #print 'len train', len(train_data)

    X_train, Y_train = labelize(train_data)
    X_dev, Y_dev = labelize(dev_data)

    #print 'X_train', type(X_train), X_train
    # print 'Y_train', Y_train.shape, Y_train
    
    # X_train = np.array([[5, 4], [1,1]])
    # Y_train = np.array([[4, 3], [1,2]])

    hdim = 10
    wdim = 10
    batch_size = 5
    n_epochs = 30
    
    
    ed = EncDec(vdim, hdim, wdim, outdim)
    
    #ed.grad_check(X_train[:10], Y_train[:10])
    #ed.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=X_dev, Y_dev=Y_dev, verbose=True)
    #ed.save_model('models/ed_simple.p')

    ed.load_model('models/ed_simple.p')
    print '5+15=',decode(ed.generate_answer(encode('5+15', indico)), outvocab)
