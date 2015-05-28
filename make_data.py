#!/usr/bin/env python

import pickle

import numpy as np

def simple_example():
    first = np.random.randint(1000)
    second = np.random.randint(1000)
    x = str(first) + ' + ' + str(second)
    y = str(first + second)
    return (x,y)

def very_simple_example():
    first = np.random.randint(100)
    second = np.random.randint(100)
    x = str(first) + '+' + str(second)
    y = str(first + second)
    return (x,y)

def simple_discr_example():
    # Come up with an (x,y) pair that doesn't match
    correct = np.random.random() < .5

    if correct:
        x,y = simple_example()
        return (x+' = '+y, 1)
    else:
        x, y1 = simple_example()
        y2 = simple_example()[1]
        return (x+' = '+y2, int(y1 == y2))


def generate_examples():
    n_train = 10000
    n_dev = 2000
    n_test = 2000

    train_file = 'data/train.txt'
    dev_file = 'data/dev.txt'
    test_file = 'data/test.txt'

    train_dat = [simple_example() for _ in xrange(n_train)]

    dev_dat_raw = [simple_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [simple_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]
    
    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated positive examples of sizes', len(train_dat), len(dev_dat), len(test_dat)


def generate_discr_examples():
    n_train = 10000
    n_dev = 2000
    n_test = 2000

    train_file = 'data/neg_train.txt'
    dev_file = 'data/neg_dev.txt'
    test_file = 'data/neg_test.txt'

    train_dat = [simple_discr_example() for _ in xrange(n_train)]

    dev_dat_raw = [simple_discr_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [simple_discr_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]
    
    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated discriminative data of sizes', len(train_dat), len(dev_dat), len(test_dat)

def generate_easy_examples():
    """Two-digit addition, no spaces"""
    n_train = 500
    n_dev = 100
    n_test = 100

    train_file = 'data/2dig_train.p'
    dev_file = 'data/2dig_dev.p'
    test_file = 'data/2dig_test.p'

    train_dat = [very_simple_example() for _ in xrange(n_train)]

    dev_dat_raw = [very_simple_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [very_simple_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]

    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated positive examples of sizes', len(train_dat), len(dev_dat), len(test_dat)


    

if __name__ == '__main__':
    # generate_examples()
    # generate_discr_examples()
    generate_easy_examples()
