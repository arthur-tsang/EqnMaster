#!/usr/bin/env python

from rnnlm import RNNLM
from misc import lengthen, get_data
import sys
import pickle
import numpy as np

class NaiveRnnlm:

    def __init__(self, scramble_name = 'two_dig_scramble', bptt = 1):

        # for now, sort of cheat and assume fixed size inputs and outputs
        self.x_len = 3
        self.y_len = 4

        self.alpha = .1
        self.n_epochs = 40

        self.hdim = 50
        self.vocab = list('0123456789+ ') # list of all possible characters we might see
        self.vdim = len(self.vocab)
        self.vocabmap = {char:i for i,char in enumerate(self.vocab)} # map char to idx number

        self.rnns = [RNNLM(np.zeros((self.vdim, self.hdim)), bptt = bptt) for _ in range(self.y_len)]

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
    def unscrambled_simple(self, x_string, i):
        return ''.join(c for c in self.lengthen_double(x_string) if c != ' ' and c != '+')
    def scramble_simple(self, x_string, i):
        return self.scramble_double(x_string)
    def two_dig_scramble(self, x_string, i):
        # where i is the output digit we're computing
        # in my opinion, this function knows a little too much about how to pick our digits
        x_slice = slice(0, 2) if i == 0 else slice(2*(i-1), 2*i)
        return self.scramble_double(x_string)[x_slice]
    def rot_scramble(self, x_string, i):
        six_digs = self.scramble_double(x_string)
        start_dig = 0 if i == 0 else i - 1
        return [c for c in reversed(six_digs[start_dig:] + six_digs[:start_dig])]
    def rot_scramble_half(self, x_string, i):
        return self.rot_scramble(x_string, i)[3:]
            

    def train(self, xy_data, rnns = None):
        # This function trains one RNN for each possible output digit

        self.rnns = rnns if rnns is not None else self.rnns

        # xs = [np.array(self.encode_expr(self.scramble_double(x))) for x,y in xy_data]
        ys = [self.encode_expr(lengthen(y, self.y_len)) for x,y in xy_data]

        # for printing purposes only
        dev_data = get_data('data/dev.txt')

        for i,rnn_i in enumerate(self.rnns):
            # where i is the index of the rnn we're using
            print 'i',i

            xs_i = [np.array(self.encode_expr(self.scramble(x, i))) for x,y in xy_data]
            ys_i = [y[i] for y in ys]
            dev_xs_i = [np.array(self.encode_expr(self.scramble(x, i))) for x,y in dev_data]
            dev_ys_i = [self.encode_expr(lengthen(y, self.y_len))[i] for x,y in dev_data]

            rnn_i.grad_check(dev_xs_i[0], dev_ys_i[0])

            for j in xrange(self.n_epochs):
                for x,y in zip(xs_i, ys):
                    rnn_i.train_point_sgd(x, y[i], self.alpha)
                # print 'train loss', rnn_i.compute_loss(xs_i, ys_i)
                if j % 10 == 0:
                    print 'dev loss', rnn_i.compute_loss(dev_xs_i, dev_ys_i)

            
            # # extra stuff to print
            # for x,y in zip(xs_i,ys)[:5]:
            #     yhat = rnn_i.predict(x)
            #     print x, yhat, np.argmax(yhat)

        return self.rnns


    def predict_one(self, x, rnns = None):
        rnns = rnns if rnns is not None else self.rnns
        if rnns is None:
            raise Exception('Model not trained!')


        x_encoded = lambda i : self.encode_expr(self.scramble(x, i))
        return ''.join(self.decode([np.argmax(rnns[i].predict(x_encoded(i))) for i in range(self.y_len)]))


if __name__ == '__main__':
    # Possible arguments are 'train', 'retrain'. Default mode is demo

    rnns_file = 'rnn_naive.txt'

    train_data = get_data('data/train.txt')

    nr = NaiveRnnlm(scramble_name = 'unscrambled_simple', bptt = 1)

    should_retrain = 'retrain' in sys.argv[1:]
    should_train = 'train' in sys.argv[1:] or should_retrain

    if should_train:
        if should_retrain:
            with open(rnns_file, 'r') as g:
                nr.rnns = pickle.load(g)

        rnns = nr.train(train_data)
        
        with open(rnns_file, 'w') as f:
            pickle.dump(rnns, f)

    else:
        with open(rnns_file, 'r') as f:
            rnns = pickle.load(f)

    
    print '123 + 456 =', nr.predict_one('123 + 456', rnns)
    print '998 + 456 =', nr.predict_one('777 + 999', rnns)
    print '9 + 9 =', nr.predict_one('9 + 9', rnns)


# With the semi-oracle scramble, one time
# 123 + 456 = 0979
# 998 + 456 = 1071
# 9 + 9 = 0418
# training a second round
# 123 + 456 = 0579
# 998 + 456 = 1344
# 9 + 9 = 0108
