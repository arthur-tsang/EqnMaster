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
