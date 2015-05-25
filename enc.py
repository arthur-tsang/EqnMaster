#!/usr/bin/env python

# We're going to do this in two separate files
# The decoder is just rnn.py?

# But this is the encoder!

# get it to learn the individual characters

# softmax and CE loss at each time-step

from nn.math import softmax, make_onehot
from misc import random_weight_matrix

class Encoder:

    def __init__(self, vdim, hdim, wdim, outdim,
                 alpha=0.005, lamb = .0001, rseed=10, bptt=1):

        # Dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.outdim = outdim
        self.wdim = wdim

        # Parameters
        random.seed(rseed)
        sigma = .1

        self.L = random.normal(0, sigma, (wdim, vdim)) # "wide" array
        self.Wh = random_weight_matrix(hdim, hdim)
        self.Wx = random_weight_matrix(hdim, wdim)
        self.b1 = np.zeros(hdim)
        self.U = random_weight_matrix(outdim, hdim)
        self.b2 = np.zeros(outdim)

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.lamb = lamb

        # Store hs and yhats
        self.hs = None
        self.yhats = None

        # grads
        self.grads = {}

    def f_prop(self, xs, ys):
        """Given a series of xs and a series of ys, returns hidden vector at
        end, and also the cost"""
        N = len(xs) # total num timesteps

        self.hs = np.zeros(self.hdim, N+1)
        self.yhats = np.zeros(self.outdim, N)

        cost = 0

        for t in xrange(N):
            Lx = self.L[:,xs[t]]
            h_prev = self.hs[:,t-1]
            z_1 = np.dot(self.Wh, h_prev) + np.dot(self.Wx, Lx) + self.b1
            h1 = np.maximum(z_1, 0)
            self.hs[:,t] = h1
            yhat = softmax(np.dot(self.U, h1) + self.b2)
            self.yhats[:,t] = yhat
            cost += -np.log(yhat[ys[t]])

        return (self.hs[N-1], cost)
            
    def b_prop(self, xs, ys, delta_decoder):
        N = len(xs)

        self.grads['U'] = np.zeros(self.U.shape)
        self.grads['Wx'] = np.zeros(self.Wx.shape)
        self.grads['Wh'] = np.zeros(self.Wh.shape)
        self.grads['b1'] = np.zeros(self.b1.shape)
        self.grads['b2'] = np.zeros(self.b2.shape)
        self.grads['L'] = np.zeros(self.L.shape)


        delta_above = delta_decoder
        for t in xrange(N-1,-1, -1):
            delta_3 = self.yhats[:,t] - make_onehot(ys[t], self.outdim)
            self.grads['U'] += np.outer(delta3, self.hs[:,t])
            self.grads['b2'] += delta_3
            dh = np.dot(np.transpose(self.U), delta_3) + delta_above
            delta_2 = dh * (self.hs[:,t] > 0)
            self.grads['b1'] += delta_2
            self.grads['Wh'] += np.outer(delta_2, self.hs[:,t-1])
            self.grads['Wx'] += np.outer(delta_2, self.L[:,xs[t]])
            self.grads['L'][:,xs[t]] += np.dot(np.transpose(self.Wx), delta_2)
            delta_below = np.dot(np.transpose(self.Wh), delta_2)

            delta_above = delta_below

    def sgd(batch, n_epochs):
        for _ in n_epochs:
            pass # TODO: implement
