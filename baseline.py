#!/usr/bin/env python

# Baseline.py
# -----------
# Here, we present a baseline roughly inspired by the bigram model

import pickle

def lengthen(string, len_full):
    assert(len(string) <= len_full)
    len_front = len_full - len(string)
    return '0' * len_front + string

def learn(xy_data):
    # In this part, we're cheating a little by assuming some things
    # about our input data that won't always be true necessarily
    x_len = 3
    y_len = 4

    caches = [{} for _ in range(x_len)]

    for x,y in xy_data:
        args = [lengthen(s, x_len) for s in x.split(' + ')]
        y = lengthen(y, y_len)
        print args, y
        for i,(x0,x1) in enumerate(zip(args[0], args[1])):
            pass
            # something about adding this to the caches


if __name__ == '__main__':
    with open('data/train.txt', 'r') as f:
        mydata = pickle.load(f)
        
    x_first = mydata[0][0]
    learn(mydata)


