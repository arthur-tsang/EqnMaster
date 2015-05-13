#!/usr/bin/env python

# Baseline.py
# -----------
# Here, we present a baseline roughly inspired by the bigram model

from collections import Counter, defaultdict
from misc import lengthen, get_data

class BigramBaseline:

    def __init__(self):

        # We're cheating a little by assuming some things about our input
        # data that won't always be true necessarily.
        self.x_len = 3
        self.y_len = 4

        self.cache = None

    def num_split(self, x_string):
        # split args by operator, e.g. '21 + 12' => ['021','012']
        # assuming only addition for now
        return [lengthen(s, self.x_len) for s in x_string.split(' + ')]

    def learn(self, xy_data):
        # This function is supposed to learn some stats from the training
        # data and store it in our cache

        # Structure of Cache:
        # (index of x0 and x1, digit x0 of first arg, digit x1 of second arg, index of answer)
        # the counter then stores how many times we see each character
        #   appear for this digit of the answer
        self.cache = defaultdict(Counter)

        for x,y in xy_data:
            # we split our args by the operator
            nums = self.num_split(x)
            y = lengthen(y, self.y_len)
            for i,(x0,x1) in enumerate(zip(nums[0], nums[1])): # for digit of x0,x1
                for j,y0 in enumerate(y): # for digit of y
                    self.cache[(i,x0,x1,j)][y0] += 1

        return self.cache

    def predict_one(self, x, cache = None):
        # predicts one answer given stats cache and input

        cache = cache if cache is not None else self.cache
        if cache is None:
            raise Exception('Model not trained -- needs to cache data values')

        y_counts = [Counter() for _ in range(self.y_len)]
    
        nums = self.num_split(x)
        for i,(x0,x1) in enumerate(zip(nums[0], nums[1])): # for x digit
            for j in xrange(self.y_len): # for y digit
                y_counts[j] += cache[(i,x0,x1,j)]

        best_answer = ''.join([c.most_common(1)[0][0] for c in y_counts])

        return best_answer
    


if __name__ == '__main__':
    train_data = get_data('data/train.txt')
        
    bb = BigramBaseline()

    cache = bb.learn(train_data)
    print '123 + 456 =', bb.predict_one('123 + 456')
    print '998 + 456 =', bb.predict_one('998 + 456')
    print '9 + 9 =', bb.predict_one('9 + 9')
