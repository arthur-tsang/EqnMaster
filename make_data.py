#!/usr/bin/env python

import pickle

import numpy as np

def simple_example():
    first = np.random.randint(1000)
    second = np.random.randint(1000)
    x = str(first) + ' + ' + str(second)
    y = str(first + second)
    return (x,y)


def generate_examples():
    n_train = 10000
    n_dev = 2000
    n_test = 2000

    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    test_file = 'data/test.txt'

    train_dat = [simple_example() for _ in xrange(n_train)]
    dev_dat = [simple_example() for _ in xrange(n_dev)]
    test_dat = [simple_example() for _ in xrange(n_test)]
    
    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated data'

if __name__ == '__main__':
    generate_examples()
