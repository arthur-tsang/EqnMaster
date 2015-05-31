# We're going to do this in two separate files
# The decoder is just rnn.py?

# But this is the encoder!

# get it to learn the individual characters

# softmax and CE loss at each time-step

import pickle
import numpy as np
from nn.math import softmax, make_onehot
from misc import random_weight_matrix

class Encoder:

    def __init__(self, vdim, hdim, wdim,
                 alpha=0.005, rho = 0.0001, rseed=10):

        # Dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.vdim = vdim
        self.wdim = wdim

        # Parameters
        np.random.seed(rseed)
        sigma = .1

        self.params = {}
        self.params['L'] = np.random.normal(0, sigma, (wdim, vdim)) # "wide" array
        self.params['Wh'] = random_weight_matrix(hdim, hdim)
        self.params['Wx'] = random_weight_matrix(hdim, wdim)
        self.params['b1'] = np.zeros(hdim)
        # self.params['U'] = random_weight_matrix(vdim, hdim)
        # self.params['b2'] = np.zeros(vdim)

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho

        # Store hs and yhats
        self.hs = None
        self.yhats = None

        # grads
        self.grads = {}

    def f_prop(self, xs):
        """Given a series of xs and a series of ys, returns hidden vector at
        end, and also the cost"""
        N = len(xs) # total num timesteps
        L = self.params['L']
        Wh = self.params['Wh']
        Wx = self.params['Wx']
        # U = self.params['U']
        b1 = self.params['b1']
        # b2 = self.params['b2']

        self.hs = np.zeros([self.hdim, N+1])
        # self.yhats = np.zeros([self.vdim, N])
        cost = 0

        for t in xrange(N):
            Lx = L[:,xs[t]]
            h_prev = self.hs[:,t-1]
            z_1 = np.dot(Wh, h_prev) + np.dot(Wx, Lx) + b1
            h1 = np.maximum(z_1, 0)
            self.hs[:,t] = h1
            # yhat = softmax(np.dot(U, h1) + b2)
            # self.yhats[:,t] = yhat
            # cost += -np.log(yhat[ys[t]])
        return self.hs[:, N-1]
            
    def b_prop(self, xs, delta_decoder):

        L = self.params['L']
        Wh = self.params['Wh']
        Wx = self.params['Wx']
        # U = self.params['U']
        b1 = self.params['b1']
        # b2 = self.params['b2']
        N = len(xs)

        delta_above = delta_decoder
        for t in xrange(N-1,-1, -1):
            # delta_3 = self.yhats[:,t] - make_onehot(ys[t], self.vdim)
            # self.grads['U'] += np.outer(delta_3, self.hs[:,t])
            # self.grads['b2'] += delta_3
            # dh = np.dot(np.transpose(U), delta_3) + delta_above
            dh = delta_above
            delta_2 = dh * (self.hs[:,t] > 0)
            self.grads['b1'] += delta_2
            self.grads['Wh'] += np.outer(delta_2, self.hs[:,t-1])
            self.grads['Wx'] += np.outer(delta_2, L[:,xs[t]])
            self.grads['L'][:,xs[t]] += np.dot(np.transpose(Wx), delta_2)
            delta_below = np.dot(np.transpose(Wh), delta_2)

            delta_above = delta_below



    def process_batch(self, all_xs):

        for key in self.params:
            self.grads[key] = np.zeros(self.params[key].shape)

        # Processes a single batch of input data; returns
        # the average cost, and updates gradients.
        tot_cost = 0.0
        batch_size = len(all_xs)
        for xs in all_xs:
            self.f_prop(xs)
            self.b_prop(xs, np.zeros(self.hdim))

        # Compute average cost
        avg_cost = tot_cost/batch_size
        avg_cost += 0.5*self.rho*(np.sum(self.params['Wh']**2) + np.sum(self.params['Wx']**2))

        # Compute average grads
        for key in self.grads:
            self.grads[key] /= batch_size

        # Add regularization to grads
        self.grads['Wh'] += self.rho*self.params['Wh']
        self.grads['Wx'] += self.rho*self.params['Wx']
        # self.grads['U'] += self.rho*self.params['U']
        # print 'Avg Cost:', avg_cost
        return avg_cost


    def regularize(self):
        self.grads['Wh'] += self.rho*self.params['Wh']
        self.grads['Wx'] += self.rho*self.params['Wx']
        # self.grads['U'] += self.rho*self.params['U']

        # reg_cost = 0.5*self.rho*(np.sum(self.params['Wh']**2) + np.sum(self.params['Wx']**2) + np.sum(self.params['U']**2))
        reg_cost = 0.5*self.rho*(np.sum(self.params['Wh']**2) + np.sum(self.params['Wx']**2))
        return reg_cost

    def update_parameters(self):
        for key in self.params:
            self.params[key] += -1*self.alpha*self.grads[key]
 

    def divide_grads(self, batch_size):
        # For batch training, divide all grad-sums by batch-size
        for key in self.grads:
            self.grads[key] /= float(batch_size)

        
    # def sgd(self, batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True):
    #     # Implentation of SGD over all training data

    #     N = len(X_train)
    #     iterations_per_epoch = N / batch_size # using SGD

    #     # 1 epoch is 1 pass over training data
    #     for epoch in xrange(n_epochs):

    #         # For every sub-iteration
    #         for i in xrange(iterations_per_epoch):

    #             # Sample a batch
    #             batch_mask = np.random.choice(N, batch_size)
    #             X_batch = X_train[batch_mask]
    #             Y_batch = Y_train[batch_mask]
    #             avg_cost = self.process_batch(X_batch, Y_batch)

    #             # Update with SGD
    #             for key in self.params:
    #                 self.params[key] += -1*self.alpha*self.grads[key]

    #         # Print progress
    #         if verbose:
    #             print "Epoch", epoch
    #             print "Training Cost:", self.process_batch(X_train, Y_train)
    #             print "Dev Cost:", self.process_batch(X_dev, Y_dev)

    
    def grad_check(self, X):
        h = 1e-5
        X = np.ndarray.astype(X, np.float)
        self.process_batch(X)
        grads = dict(self.grads)
        for key in self.params:
            print 'Gradient check for ', key
            it = np.nditer(self.params[key], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                old_val = it[0].copy()
                it[0] = old_val - h
                low_cost = self.process_batch(X)
                it[0] = old_val + h
                high_cost = self.process_batch(X)
                it[0] = old_val
                num_grad = float(high_cost - low_cost)/(2*h)
                diff = grads[key][it.multi_index] - num_grad
                if abs(diff) > 1e-4:
                    print "Grad Check Failed -- error:", diff
                it.iternext()
        print "Grad Check Finished!"

    # Save model (hyper)parameters to a file
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump((self.params, self.alpha, self.rho), f)

    # Load model (hyper)parameters to a file
    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            self.params, self.alpha, self.rho = pickle.load(f)





