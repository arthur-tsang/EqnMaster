from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = (L0.shape if U0 is None else U0.shape))
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        self.alpha = alpha

        self.bptt = bptt

        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        random.seed(rseed)
        
        sigma = sqrt(0.1)
        self.sparams.L = random.normal(0, sigma, L0.shape)
        self.params.U = random.normal(0, sigma, param_dims['U'])
        
        # Initialize H matrix, as with W and U in part 1
        self.params.H = random_weight_matrix(*param_dims['H'])

        self.lamb = .0001 # regularization
        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H)
                and self.sgrads (for L,U)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        #ps = zeros((ns, self.vdim))
        yhat = None

        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        #hs[-1] is kindly all 0s (always)
        for t in xrange(ns):
            theta_t = dot(self.params.H, hs[t-1]) + self.sparams.L[xs[t]]
            hs[t] = sigmoid(theta_t)
            if t == ns - 1:
                yhat = softmax(dot(self.params.U, hs[t]))

        ##
        # Backward propagation through time
        def get_delta_i(delta_next, t):
            ddht = dot(transpose(self.params.H), delta_next)
            return ddht * hs[t] * (1- hs[t])

        #for t in xrange(ns-1, -1, -1):
        t = ns-1

        dJ_dUht = yhat
        dJ_dUht[ys] -= 1 # (-y + yhat)

        self.grads.U += outer(dJ_dUht, hs[t])
        dJ_dht = dot(transpose(self.params.U), dJ_dUht) # h(t) = sig(theta)
        dJ_dThetat = dJ_dht * (hs[t]) * (1 - hs[t])
        delta_t = dJ_dThetat
            
        # BPTT
        delta_next = None
        i = t
        while i >= max(t - self.bptt + 1, 0):
            # note that bptt=1 means we only run it on regular t
            delta_i = get_delta_i(delta_next, i) if i != t else delta_t
            self.sgrads.L[xs[i]] = delta_i
            self.grads.H += outer(delta_i, hs[i-1])

            delta_next = delta_i
            i -= 1


        # regularization
        self.grads.H += self.lamb * self.params.H
        self.grads.U += self.lamb * self.params.U

        #### END YOUR CODE ####

    def predict(self, xs):
        # predicts yhat based on xs

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        #ps = zeros((ns, self.vdim))
        yhat = None

        ##
        # Forward propagation
        #hs[-1] is kindly all 0s (always)
        for t in xrange(ns):
            theta_t = dot(self.params.H, hs[t-1]) + self.sparams.L[xs[t]]
            hs[t] = sigmoid(theta_t)
            if t == ns - 1:
                yhat = softmax(dot(self.params.U, hs[t]))

        return yhat


    # def forward_prop(self, xs, ys, H = self.params.H, L = self.sparams.L, U = self.params.U):
    #     # forward prop helper function
    #     # for a grad_check algorithm

    #     # Expect xs as list of indices
    #     ns = len(xs)

    #     # make matrix here of corresponding h(t)
    #     # hs[-1] = initial hidden state (zeros)
    #     hs = zeros((ns+1, self.hdim))
    #     # predicted probas
    #     #ps = zeros((ns, self.vdim))
    #     yhat = None

    #     ##
    #     # Forward propagation
    #     #hs[-1] is kindly all 0s (always)
    #     for t in xrange(ns):
    #         theta_t = dot(H, hs[t-1]) + L[xs[t]]
    #         hs[t] = sigmoid(theta_t)
    #         if t == ns - 1:
    #             yhat = softmax(dot(U, hs[t]))

    #     return yhat


    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(x)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        ns = len(xs)
        h_prev = zeros(self.hdim)
        for t in xrange(ns):
            h_t = sigmoid(dot(self.params.H, h_prev) + self.sparams.L[xs[t]])
            if t == ns - 1:
                yhat_t = softmax(dot(self.params.U, h_t))
                J = -log(yhat_t[ys])

            h_prev = h_t

        J += .5 * self.lamb * (sum(self.params.H**2) + sum(self.params.U**2))

        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in zip(X, Y)]) # used to be itertools.izip

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


