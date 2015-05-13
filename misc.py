##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    eps = sqrt(6.) / sqrt(m + n)
    A0 = random.uniform(-eps, eps, (m,n)) # (low, high, size)
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0

def lengthen(num_string, len_full):
    # So that our number strings are more uniform, we can lengthen all
    # numbers by adding zeros at the front
    assert(len(num_string) <= len_full)
    len_front = len_full - len(num_string)
    return '0' * len_front + num_string
