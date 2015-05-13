#!/usr/bin/env python

from rnnlm import RNNLM
from misc import lengthen, get_data
import pickle
import numpy as np

class NaiveRnnlm:

    def __init__(self):

        # for now, sort of cheat and assume fixed size inputs and outputs
        self.x_len = 3
        self.y_len = 4

        self.alpha = .1

        self.hdim = 5
        self.vocab = list('0123456789+ ') # list of all possible characters we might see
        self.vdim = len(self.vocab)
        self.vocabmap = {char:i for i,char in enumerate(self.vocab)} # map char to idx number

        self.rnns = None

    def encode_expr(self, expr):
        return [self.vocabmap[c] for c in expr]
    def decode(self, indices):
        return ''.join([self.vocab[idx] for idx in indices])

    def lengthen_double(self, x_string):
        # format from '21 + 12' -> '021 + 012'
        return ' + '.join([lengthen(s, self.x_len) for s in x_string.split(' + ')])
    def scramble_double(self, x_string):
        # format 'abc + 123' to 'a1b2c3'
        nums = x_string.split(' + ')
        return ''.join([x1 + x2 for x1,x2 in zip(nums[0], nums[1])])

    def train(self, xy_data):
        # This function trains one RNN for each possible output digit

        self.rnns = [RNNLM(np.zeros((self.vdim, self.hdim))) for _ in range(self.y_len)]

        n_epochs = 20

        xs = [np.array(self.encode_expr(self.scramble_double(x))) for x,y in xy_data]
        ys = [self.encode_expr(lengthen(y, self.y_len)) for x,y in xy_data]

        for i,rnn_i in enumerate(self.rnns):
            # where i is the index of the rnn we're using
            print 'i',i

            ys_i = [y[i] for y in ys]

            for _ in xrange(n_epochs):
                for x,y in zip(xs, ys):
                    rnn_i.train_point_sgd(x, y[i], self.alpha)
                print 'loss', rnn_i.compute_loss(xs, ys_i)

            # extra stuff to print
            # for x,y in zip(xs,ys)[:5]:
            #     yhat = rnn_i.predict(x)
            #     print self.decode(x), yhat, np.argmax(yhat)

        return self.rnns


    def predict_one(self, x, rnns = None):
        rnns = rnns if rnns is not None else self.rnns
        if rnns is None:
            raise Exception('Model not trained!')

        return ''.join(self.decode([np.argmax(rnns[i].predict(self.encode_expr(self.scramble_double(x)))) for i in range(self.y_len)]))


if __name__ == '__main__':
    train_data = get_data('data/train.txt')

    nr = NaiveRnnlm()

    rnns = nr.train(train_data)
    
    with open('rnn_naive.txt', 'w') as f:
        pickle.dump(rnns, f)

    # with open('rnn_naive.txt', 'r') as f:
    #     rnns = pickle.load(f)

    print '123 + 456 =', nr.predict_one('123 + 456', rnns)
    print '998 + 456 =', nr.predict_one('777 + 999', rnns)
    print '9 + 9 =', nr.predict_one('9 + 9', rnns)
