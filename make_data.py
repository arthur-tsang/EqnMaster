#!/usr/bin/env python

import pickle

import numpy as np

# TODO: sort the functions in a way that makes sense

def composition_example(operators=['+', '-', '*']):
    def randnr():
        return str(np.random.randint(1000))
    def randop():
        return np.random.choice(operators)

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

def simpcomp_example():
    return composition_example(['+','-'])

def add_ndigs_example(ndigs, operator='+'):
    # addition of exactly n digits by n digits
    assert(ndigs > 0)
    assert(type(operator) == type('+'))
    def num_gen():
        return str(np.random.randint(9) + 1) + ''.join(str(np.random.randint(10)) for _ in range(ndigs-1))
    full_string = num_gen() + operator + num_gen()
    assert(len(full_string) == 2 * ndigs + 1)
    return (full_string, str(eval(full_string)))


def simple_example(operator = ' + '):
    assert(type(operator) == type(' '))
    first = np.random.randint(1000)
    second = np.random.randint(1000)
    x = str(first) + operator + str(second)
    y = str(eval(x))
    return (x,y)



def very_simple_example():
    first = np.random.randint(100)
    second = np.random.randint(100)
    x = str(first) + '+' + str(second)
    y = str(eval(x))
    return (x,y)

def mult_example():
    return simple_example('*')

def subtr_example():
    return simple_example('-')

def simple_discr_example():
    # Come up with an (x,y) pair that doesn't match
    return generic_discr_example(simple_example)

def subtr_discr_example():
    return generic_discr_example(subtr_example)

def mult_discr_example():
    return generic_discr_example(mult_example)

def comp_discr_example():
    return generic_discr_example(composition_example)

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
    generate_generic_example('3dig', simple_example)

def generate_discr_examples():
    generate_generic_examples('d_add', simple_discr_example)

def generate_easy_examples():
    """Two-digit addition, no spaces"""
    generate_generic_examples('2dig', very_simple_example, n_train=500, n_dev=100, n_test=100)

def generate_mult_examples():
    """Three-digit multiplication, no spaces"""
    generate_generic_examples('mult', mult_example)

def generate_subtr_examples():
    """Three-digit subtraction, no spaces"""
    generate_generic_examples('subtr', subtr_example)

def generate_subtr_discr_examples():
    generate_generic_examples('d_subtr', subtr_discr_example)

def generate_mult_discr_examples():
    generate_generic_examples('d_mult', mult_discr_example)

def triple_pickle(filenames, datasets):
    # Just a convenience function to help you pickle data
    for filename, dat in zip(filenames, datasets):
        assert(len(filename) > 1)
        with open(filename, 'wb') as f:
            pickle.dump(dat, f)
    

def generate_simpcomp_examples():
    generate_generic_examples('simpcomp', simpcomp_example)
    print 'generated simpcomp (comp with only + and -) examples'

def generate_composition_examples():
    generate_generic_examples('comp', composition_example, n_train = 500000)
    print 'generated composition examples'
    
def generate_discr_composition_examples():
    generate_generic_examples('d_comp', comp_discr_example)
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

def generate_ndig_subtr_examples():
    generate_generic_examples('4subtr', lambda: add_ndigs_example(4, operator='-'))
    generate_generic_examples('5subtr', lambda: add_ndigs_example(5, operator='-'))
    generate_generic_examples('6subtr', lambda: add_ndigs_example(6, operator='-'))
    generate_generic_examples('7subtr', lambda: add_ndigs_example(7, operator='-'))

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
    #generate_ndig_examples()
    # generate_generic_examples('bigmult', mult_example, n_train=100000)
    # # print mult_discr_example()

    # generate_discr_composition_examples()

    # generate_ndig_subtr_examples()
    generate_simpcomp_examples()
