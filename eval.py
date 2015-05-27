#!/usr/bin/env python

# Here, we evaluate the models according to the dev sets

# This file is temporarily closed for renovation (metaphorically speaking

from naive_rnnlm import NaiveRnnlm
from naive_rnnlm_discr import NaiveRnnlmDiscr
from baseline import BigramBaseline
from misc import lengthen, get_data
import os.path
import pickle

def bool_metric(correct, given):
    return int(correct == given)

def dig_metric(correct, given):
    # Evaluation metric: 1/4 point per matching digit
    # (not strict booleans here, since we're not good enough (yet))
    y_len = 4
    correct = lengthen(correct, y_len) # add initial zeros
    given = lengthen(given, y_len)
    score = sum(c == g for c,g in zip(correct, given)) / float(y_len)
    return score

def eval_model(predict_fn, xy_data, metric = dig_metric):
    scores = [metric(y, predict_fn(x)) for x,y in xy_data]
    return 1.0 * sum(scores) / len(scores)

def nr_test(rnns_file, data, discr = False):
    print 'Warning: this function might not act right'
    metric = dig_metric if not discr else bool_metric
    nr = NaiveRnnlm() if not discr else NaiveRnnlmDiscr()
    if os.path.exists(rnns_file):
        with open(rnns_file, 'r') as f:
            nr.rnns = pickle.load(f)
        print 'nr at', rnns_file, eval_model(nr.predict_one, data, metric)
    else:
        print 'nr at', rnns_file, 'not found'

if __name__ == '__main__':
    train_data = get_data('data/train.txt')
    dev_data = get_data('data/dev.txt')
    train_data_discr = get_data('data/neg_train.txt')
    dev_data_discr = get_data('data/neg_dev.txt')

    


    # print 'Dev set scores'

    # # bigram baseline part
    # bb = BigramBaseline()
    # bb.learn(train_data)
    # print 'bb eval:', eval_model(bb.predict_one, dev_data)
    
    # # naive rnn part
    # nr_test('rnn_naive.txt', dev_data)
    # nr_test('rnn_naive_oracle.txt', dev_data)
    # nr_test('rnn_naive_oracle_bptt.txt', dev_data)
    # nr_test('rnn_naive_rot.txt', dev_data)
    # nr_test('rnn_naive_rot_bptt.txt', dev_data)
    # nr_test('rnn_naive_discr.txt', dev_data_discr, True)


    # print 'Train set scores'
    # # bigram baseline part
    # print 'bb eval:', eval_model(bb.predict_one, train_data)
    
    # # naive rnn part
    # nr_test('rnn_naive.txt', train_data)
    # nr_test('rnn_naive_oracle.txt', train_data)
    # nr_test('rnn_naive_oracle_bptt.txt', train_data)
    # nr_test('rnn_naive_rot.txt', train_data)
    # nr_test('rnn_naive_rot_bptt.txt', train_data)
    # nr_test('rnn_naive_discr.txt', train_data_discr, True)
