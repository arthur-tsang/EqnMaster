#!/usr/bin/env python

# Here, we evaluate the models according to the dev sets

from naive_rnnlm import NaiveRnnlm
from baseline import BigramBaseline
from misc import lengthen, get_data
import pickle

def metric(correct, given):
    # Evaluation metric: 1/4 point per matching digit
    # (not strict booleans here, since we're not good enough (yet))
    y_len = 4
    correct = lengthen(correct, y_len) # add initial zeros
    given = lengthen(given, y_len)
    score = sum(c == g for c,g in zip(correct, given)) / float(y_len)
    return score

def eval_model(predict_fn, xy_data):
    scores = [metric(y, predict_fn(x)) for x,y in xy_data]
    return 1.0 * sum(scores) / len(scores)

if __name__ == '__main__':
    train_data = get_data('data/train.txt')
    dev_data = get_data('data/dev.txt')
    
    # bigram baseline part
    bb = BigramBaseline()
    bb.learn(train_data)
    print 'bb eval:', eval_model(bb.predict_one, dev_data)
    
    # naive rnn part
    nr = NaiveRnnlm()
    # skipping learning because that takes a bit of time
    with open('rnn_naive.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr eval:', eval_model(nr.predict_one, dev_data)

    # semi-oracle rnn part
    with open('rnn_naive_oracle.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr semi-oracle eval:', eval_model(nr.predict_one, dev_data)

    # semi-oracle rnn with bptt part
    with open('rnn_naive_oracle_bptt.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr semi-oracle bptt eval:', eval_model(nr.predict_one, dev_data)

    # input-rotation rnn without bptt part
    with open('rnn_naive_rot.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr rot eval:', eval_model(nr.predict_one, dev_data)

    # input-rotation latter half rnn without bptt part
    with open('rnn_naive_rot_half.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr rot eval:', eval_model(nr.predict_one, dev_data)

    # input-rotation rnn with bptt part
    with open('rnn_naive_rot_bptt.txt', 'r') as f:
        nr.rnns = pickle.load(f)
    print 'nr rot bptt eval:', eval_model(nr.predict_one, dev_data)
