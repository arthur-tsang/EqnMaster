#!/usr/bin/env python

from rnnlm import RNNLM

import pickle
import numpy as np

alpha = .1

hdim = 5
vocab = list('0123456789+ ') # list of all possible characters we might see
vdim = len(vocab)
vocabmap = {char:i for i,char in enumerate(vocab)} # map char to idx number
def encode_expr(expr):
    return [vocabmap[c] for c in expr]
def decode(indices):
    return ''.join([vocab[idx] for idx in indices])

# for now, sort of cheat and assume fixed size inputs and outputs
x_len = 3
y_len = 4

def lengthen(num_string, len_full):
    # So that our number strings are more uniform, we can lengthen all
    # numbers by adding zeros at the front
    assert(len(num_string) <= len_full)
    len_front = len_full - len(num_string)
    return '0' * len_front + num_string

def lengthen_double(x_string):
    return ' + '.join([lengthen(s, x_len) for s in x_string.split(' + ')])

def train(xy_data):
    n_epochs = 200

    xs = [encode_expr(lengthen_double(x)) for x,y in xy_data]
    ys = [encode_expr(lengthen(y, y_len)) for x,y in xy_data]
    
    ys4 = [y[3] for y in ys]

    # for now, train to predict only last digit
    rnn4 = RNNLM(np.zeros((vdim, hdim))) # dimensions specified at top

    for _ in xrange(n_epochs):
        for x,y in zip(xs, ys):
            rnn4.train_point_sgd(x, y[3], alpha)
        #print 'things we use to compute loss:',xs[:5], ys4[:5]
        print 'loss', rnn4.compute_loss(np.array(xs), np.array(ys4))

    for x,y in zip(xs,ys)[:5]:
        yhat = rnn4.predict(x)
        print decode(x), yhat, np.argmax(yhat)



    return rnn4


if __name__ == '__main__':
    with open('data/train.txt', 'r') as f:
        train_data = pickle.load(f)

    train(train_data)
    # xs_raw = ['23 + 14']
    # ys_raw = ['37']
    #train(zip(xs_raw,ys_raw))


print 'hello'
