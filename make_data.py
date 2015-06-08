#!/usr/bin/env python

import pickle

import numpy as np

# TODO: this code style is horrible

def composition_example():
    def randnr():
        return str(np.random.randint(1000))
    def randop():
        return np.random.choice(['+','-','*'])

    a = randnr()
    b = randnr()
    c = randnr()
    d = randnr()
    op1 = randop()
    op2 = randop()
    op3 = randop()
    
    in_str = ''.join([a, op1, b, op2, c, op3, d])
    out_str = str(eval(in_str))

    return (in_str, out_str)

def add_ndigs_example(ndigs):
    # addition of exactly n digits by n digits
    assert(ndigs > 0)
    def num_gen():
        return str(np.random.randint(9) + 1) + ''.join(str(np.random.randint(10)) for _ in range(ndigs-1))
    full_string = num_gen() + '+' + num_gen()
    assert(len(full_string) == 2 * ndigs + 1)
    return (full_string, str(eval(full_string)))


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

def mult_example():
    first = np.random.randint(1000)
    second = np.random.randint(1000)
    x = str(first) + '*' + str(second)
    y = str(first * second)
    return (x,y)

def subtr_example():
    first = np.random.randint(1000)
    second = np.random.randint(1000)
    x = str(first) + '-' + str(second)
    y = str(first - second)
    return (x,y)


def simple_discr_example():
    # Come up with an (x,y) pair that doesn't match
    generic_discr_example(simple_example)

def subtr_discr_example():
    generic_discr_example(subtr_example)

def mult_discr_example():
    generic_discr_example(mult_example)

def comp_discr_example():
    generic_discr_example(composition_example)

def generic_discr_example(example_generator):
    correct = np.random.random() < .5

    if correct:
        x,y = example_generator()
        return (x+'='+y, 1)
    else:
        x, y1 = example_generator()
        y2 = example_generator()[1]
        return (x+'='+y2, int(y1 == y2))
    


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

def generate_mult_examples():
    """Three-digit multiplication, no spaces"""
    n_train = 10000
    n_dev = 2000
    n_test = 2000

    train_file = 'data/mult_train.p'
    dev_file = 'data/mult_dev.p'
    test_file = 'data/mult_test.p'

    train_dat = [mult_example() for _ in xrange(n_train)]

    dev_dat_raw = [mult_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [mult_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]

    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated positive examples of sizes', len(train_dat), len(dev_dat), len(test_dat)


def generate_subtr_examples():
    """Three-digit subtraction, no spaces"""
    n_train = 10000
    n_dev = 1000
    n_test = 1000

    train_file = 'data/subtr_train.p'
    dev_file = 'data/subtr_dev.p'
    test_file = 'data/subtr_test.p'

    train_dat = [subtr_example() for _ in xrange(n_train)]

    dev_dat_raw = [subtr_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [subtr_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]

    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated positive examples of sizes', len(train_dat), len(dev_dat), len(test_dat)    

def generate_subtr_discr_examples():
    n_train = 10000
    n_dev = 1000
    n_test = 1000

    train_file = 'data/d_subtr_train.p'
    dev_file = 'data/d_subtr_dev.p'
    test_file = 'data/d_subtr_test.p'

    train_dat = [subtr_discr_example() for _ in xrange(n_train)]

    dev_dat_raw = [subtr_discr_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [subtr_discr_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]

    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated discrimintative examples of sizes', len(train_dat), len(dev_dat), len(test_dat)    


def generate_mult_discr_examples():
    n_train = 10000
    n_dev = 2000
    n_test = 2000

    train_file = 'data/d_mult_train.p'
    dev_file = 'data/d_mult_dev.p'
    test_file = 'data/d_mult_test.p'

    train_dat = [mult_discr_example() for _ in xrange(n_train)]

    dev_dat_raw = [mult_discr_example() for _ in xrange(n_dev)]
    dev_dat = [x for x in dev_dat_raw if x not in train_dat]

    test_dat_raw = [mult_discr_example() for _ in xrange(n_test)]
    test_dat = [x for x in test_dat_raw if x not in train_dat and x not in dev_dat]
    
    with open(train_file, 'w') as f:
        pickle.dump(train_dat, f)
    with open(dev_file, 'w') as f:
        pickle.dump(dev_dat, f)
    with open(test_file, 'w') as f:
        pickle.dump(test_dat, f)

    print 'Generated discriminative data of sizes', len(train_dat), len(dev_dat), len(test_dat)

def triple_pickle(filenames, datasets):
    # Just a convenience function to help you pickle data
    for filename, dat in zip(filenames, datasets):
        assert(len(filename) > 1)
        with open(filename, 'wb') as f:
            pickle.dump(dat, f)
    

def generate_composition_examples():
    n_train = 500000
    n_dev = 2000
    n_test = 2000
    
    train_file = 'data/comp_train.p'
    dev_file = 'data/comp_dev.p'
    test_file = 'data/comp_test.p'

    n_tot = n_train + n_dev + n_test
    all_examples = set()
    while len(all_examples) < n_tot:
        if len(all_examples) % 10000 == 0:
            print 'len', len(all_examples)
        all_examples.add(composition_example())

    example_list = list(all_examples)
    train_dat = example_list[:n_train]
    dev_dat = example_list[n_train:n_train + n_dev]
    test_dat = example_list[n_train + n_dev:]

    #triple_pickle(['data/comp_train_short.p'],[train_dat[:10000]])
    triple_pickle((train_file, dev_file, test_file), (train_dat, dev_dat, test_dat))

    print 'generated composition examples'
    
def generate_discr_composition_examples():
    n_train = 10000
    n_dev = 2000
    n_test = 2000
    
    train_file = 'data/d_comp_train.p'
    dev_file = 'data/d_comp_dev.p'
    test_file = 'data/d_comp_test.p'

    n_tot = n_train + n_dev + n_test
    all_examples = set()
    while len(all_examples) < n_tot:
        if len(all_examples) % 10000 == 0:
            print 'len', len(all_examples)
        all_examples.add(composition_example())

    example_list = list(all_examples)
    train_dat = example_list[:n_train]
    dev_dat = example_list[n_train:n_train + n_dev]
    test_dat = example_list[n_train + n_dev:]

    #triple_pickle(['data/comp_train_short.p'],[train_dat[:10000]])
    triple_pickle((train_file, dev_file, test_file), (train_dat, dev_dat, test_dat))

    print 'generated discriminative composition examples'
    
def generate_generic_examples(file_short, example_generator, n_train=10000, n_dev=2000, n_test=2000):
    train_file = 'data/' + file_short + '_train.p'
    dev_file = 'data/' + file_short + '_dev.p'
    test_file = 'data/' + file_short + '_test.p'

    n_tot = n_train + n_dev + n_test

    all_examples = set()
    while len(all_examples) < n_tot:
        if len(all_examples) % 10000 == 0:
            print 'len', len(all_examples)
        all_examples.add(example_generator())

    example_list = list(all_examples)
    train_dat = example_list[:n_train]
    dev_dat = example_list[n_train:n_train + n_dev]
    test_dat = example_list[n_train + n_dev:]

    #triple_pickle(['data/comp_train_short.p'],[train_dat[:10000]])
    triple_pickle((train_file, dev_file, test_file), (train_dat, dev_dat, test_dat))

def generate_ndig_examples():
    generate_generic_examples('4dig', lambda: add_ndigs_example(4))
    generate_generic_examples('5dig', lambda: add_ndigs_example(5))
    generate_generic_examples('6dig', lambda: add_ndigs_example(6))
    generate_generic_examples('7dig', lambda: add_ndigs_example(7))

if __name__ == '__main__':
    # generate_examples()
    # generate_discr_examples()
    # generate_easy_examples()
    # generate_mult_examples()
    # generate_subtr_examples()
    # generate_subtr_discr_examples()
    # generate_mult_discr_examples()
    # generate_composition_examples()
    # generate_discr_composition_examples()
    #print composition_example()
    
    # print add_ndigs_example(5)
    generate_ndig_examples()
