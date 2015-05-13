#!/usr/bin/env python

from rnnlm import RNNLM
from misc import lengthen
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
    # format from '21 + 12' -> '021 + 012'
    return ' + '.join([lengthen(s, x_len) for s in x_string.split(' + ')])
def scramble_double(x_string):
    # format 'abc + 123' to 'a1b2c3'
    nums = x_string.split(' + ')
    return ''.join([x1 + x2 for x1,x2 in zip(nums[0], nums[1])])
def easy_double(x_string):
    # format from '23 + 456' -> '023456'
    return ''.join([lengthen(s,x_len)[2] for s in x_string.split(' + ')])

def train(xy_data):
    rnns = [RNNLM(np.zeros((vdim, hdim))) for _ in range(y_len)]

    n_epochs = 20

    xs = [np.array(encode_expr(scramble_double(x))) for x,y in xy_data]
    ys = [encode_expr(lengthen(y, y_len)) for x,y in xy_data]

    for i,rnn_i in enumerate(rnns):
        # where i is the index of the rnn we're using
        print 'i',i

        ys_i = [y[i] for y in ys]

        for _ in xrange(n_epochs):
            for x,y in zip(xs, ys):
                rnn_i.train_point_sgd(x, y[i], alpha)
            print 'loss', rnn_i.compute_loss(xs, ys_i)

        for x,y in zip(xs,ys)[:5]:
            yhat = rnn_i.predict(x)
            print decode(x), yhat, np.argmax(yhat)

    return rnns


def predict_one(rnns, x):
    return ''.join(decode([np.argmax(rnns[i].predict(encode_expr(scramble_double(x)))) for i in range(y_len)]))


if __name__ == '__main__':
    with open('data/train.txt', 'r') as f:
        train_data = pickle.load(f)

    # rnns = train(train_data)
    
    # with open('rnn_naive.txt', 'w') as f:
    #     pickle.dump(rnns, f)

    with open('rnn_naive.txt', 'r') as f:
        rnns = pickle.load(f)

    print '123 + 456 =', predict_one(rnns, '123 + 456')
    print '777 + 999 =', predict_one(rnns, '777 + 999')
