#!/usr/bin/env python

from rnnlm import RNNLM
from misc import lengthen, get_data
import sys
import pickle
import numpy as np

class NaiveRnnlmDiscr:

    def __init__(self, scramble_name = 'noscramble', bptt = 1):
        self.alpha = .1
        self.n_epochs = 100

        self.hdim = 10
        self.vocab = list('0123456789+ =') # list of all possible characters we might see
        self.vdim = len(self.vocab)
        self.vocabmap = {char:i for i,char in enumerate(self.vocab)} # map char to idx number

        self.rnn = RNNLM(np.zeros((self.vdim, self.hdim)), U0 = np.zeros((2, self.hdim)), bptt = bptt)

        self.scramble = getattr(self, scramble_name)

    def encode_expr(self, expr):
        return [self.vocabmap[c] for c in expr]
    def decode(self, indices):
        return ''.join([self.vocab[idx] for idx in indices])

    def lengthen_double(self, x_string):
        # format from '21 + 12' -> '021 + 012'
        return ' + '.join([lengthen(s, self.x_len) for s in x_string.split(' + ')])
    def scramble_double(self, x_string):
        # format 'abc + 123' to 'a1b2c3'
        lengthened = self.lengthen_double(x_string)
        nums = lengthened.split(' + ')
        return ''.join([x1 + x2 for x1,x2 in zip(nums[0], nums[1])])
    def noscramble(self, x_string):
        return x_string
    # def unscrambled_simple(self, x_string, i):
    #     return ''.join(c for c in self.lengthen_double(x_string) if c != ' ' and c != '+')
    # def scramble_simple(self, x_string, i):
    #     return self.scramble_double(x_string)
    # def two_dig_scramble(self, x_string, i):
    #     # where i is the output digit we're computing
    #     # in my opinion, this function knows a little too much about how to pick our digits
    #     x_slice = slice(0, 2) if i == 0 else slice(2*(i-1), 2*i)
    #     return self.scramble_double(x_string)[x_slice]
    # def rot_scramble(self, x_string, i):
    #     six_digs = self.scramble_double(x_string)
    #     start_dig = 0 if i == 0 else i - 1
    #     return [c for c in reversed(six_digs[start_dig:] + six_digs[:start_dig])]
    # def rot_scramble_half(self, x_string, i):
    #     return self.rot_scramble(x_string, i)[3:]
            

    def train(self, xy_data, rnn = None):
        # This function trains one RNN

        self.rnn = rnn if rnn is not None else self.rnn

        xs = [np.array(self.encode_expr(self.scramble(x))) for x,y in xy_data]
        ys = [y for x,y in xy_data]

        # for printing purposes only
        dev_data = get_data('data/neg_dev.txt')

        dev_xs = [np.array(self.encode_expr(self.scramble(x))) for x,y in dev_data]
        dev_ys = [y for x,y in dev_data]

        self.rnn.grad_check(dev_xs[0], dev_ys[0])

        for j in xrange(self.n_epochs):
            for x,y in zip(xs, ys):
                self.rnn.train_point_sgd(x, y, self.alpha)
            # print 'train loss', rnn_i.compute_loss(xs_i, ys_i)
            if j % 10 == 0:
                print 'dev loss', self.rnn.compute_loss(dev_xs[:100], dev_ys[:100]), 'train loss', self.rnn.compute_loss(xs[:100], ys[:100])

            
        # # extra stuff to print
        # for x,y in zip(xs_i,ys)[:5]:
        #     yhat = rnn_i.predict(x)
        #     print x, yhat, np.argmax(yhat)

        return self.rnn


    def predict_one(self, x, rnn = None):
        rnn = rnn if rnn is not None else self.rnn
        if rnn is None:
            raise Exception('Model not trained!')

        x_encoded = self.encode_expr(self.scramble(x))
        return np.argmax(rnn.predict(x_encoded))


if __name__ == '__main__':
    # Possible arguments are 'train', 'retrain'. Default mode is demo

    rnn_file = 'rnn_naive_discr.txt'

    train_data = get_data('data/neg_train.txt')

    nr = NaiveRnnlmDiscr(scramble_name = 'noscramble', bptt = 2)

    should_retrain = 'retrain' in sys.argv[1:]
    should_train = 'train' in sys.argv[1:] or should_retrain

    if should_train:
        if should_retrain:
            with open(rnn_file, 'r') as g:
                nr.rnn = pickle.load(g)

        rnn = nr.train(train_data)
        
        with open(rnn_file, 'w') as f:
            pickle.dump(rnn, f)

    else:
        with open(rnn_file, 'r') as f:
            rnns = pickle.load(f)
