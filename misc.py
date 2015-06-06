##
# Miscellaneous helper functions
##

from numpy import *
import pickle
import os.path
#import cPickle
import marshal

def random_weight_matrix(m, n):
    # A random-weight initialization for the matrices
    eps = sqrt(6.) / sqrt(m + n)
    A0 = random.uniform(-eps, eps, (m,n)) # (low, high, size)
    assert(A0.shape == (m,n))
    return A0

def lengthen(num_string, len_full):
    # So that our number strings are more uniform, we can lengthen all
    # numbers by adding zeros at the front
    assert(len(num_string) <= len_full)
    len_front = len_full - len(num_string)
    return '0' * len_front + num_string

def get_data(filename):
    # Returns pickled data stored at the given filename
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data

# def try_load(filename, dumper):
#     # dumper is a function that takes the filename as its sole
#     # argument (and returns a theano compiled function in our case)
#     if not os.path.isfile(filename):
#         dumper(filename)

#     with open(filename, 'rb') as f:
#         cached = pickle.load(f)

#     return cached

# def try_load(filename, constructor):
#     # First tries to load from file, then uses time-consuming constructor if unavailable
#     # Use this for theano compiled functions
#     if os.path.isfile(filename):
#         print 'using cache'
#         with open(filename, 'rb') as f:
#             cached = marshal.load(f)
#     else:
#         print 'need to construct'
#         with open(filename, 'wb') as f:
#             cached = constructor()
#             marshal.dump(cached, f)
#     return cached

