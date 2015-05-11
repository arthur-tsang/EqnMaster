#!/usr/bin/env python

# Baseline.py
# -----------
# Here, we present a baseline roughly inspired by the bigram model

from collections import Counter, defaultdict
import pickle

# We're cheating a little by assuming some things about our input
# data that won't always be true necessarily.
x_len = 3
y_len = 4

def lengthen(num_string, len_full):
    # So that our number strings are more uniform, we can lengthen all
    # numbers by adding zeros at the front
    assert(len(num_string) <= len_full)
    len_front = len_full - len(num_string)
    return '0' * len_front + num_string

def get_two_args(x_string):
    # split args by operator, e.g. '21 + 12' => ['021','012']
    # assuming only addition for now
    return [lengthen(s, x_len) for s in x_string.split(' + ')]

def learn(xy_data):
    # This function is supposed to learn some stats from the training
    # data and store it in our cache

    # Structure of Cache:
    # (index of x0 and x1, digit x0 of first arg, digit x1 of second arg, index of answer)
    # the counter then stores how many times we see each character
    #   appear for this digit of the answer
    cache = defaultdict(Counter)

    for x,y in xy_data:
        # we split our args by the operator
        args = get_two_args(x)
        y = lengthen(y, y_len)
        for i,(x0,x1) in enumerate(zip(args[0], args[1])): # for digit of x0,x1
            for j,y0 in enumerate(y): # for digit of y
                cache[(i,x0,x1,j)][y0] += 1

    return cache

def predict_one(cache, x):
    # predicts one answer given stats cache and input

    y_counts = [Counter() for _ in range(y_len)]
    
    args = get_two_args(x)
    for i,(x0,x1) in enumerate(zip(args[0], args[1])):
        for j in xrange(y_len):
            y_counts[j] += cache[(i,x0,x1,j)]

    best_answer = ''.join([c.most_common(1)[0][0] for c in y_counts])

    return best_answer
    


if __name__ == '__main__':
    with open('data/train.txt', 'r') as f:
        train_data = pickle.load(f)
        
    cache = learn(train_data)
    print predict_one(cache, '123 + 456')
    print predict_one(cache, '998 + 456')
    print predict_one(cache, '9 + 9')


