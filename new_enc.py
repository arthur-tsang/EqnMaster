import numpy as np
import theano.tensor as T
from theano import function, shared, scan
from theano.tensor.nnet import sigmoid, softmax

from misc import random_weight_matrix

rng = np.random

class NewEnc:

    def __init__(self, vdim, hdim, wdim, alpha=.005, rho=.0001, rseed=10):
        
        # Dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.vdim = vdim
        self.wdim = wdim

        # Parameters
        np.random.seed(rseed)

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho


        ## Theano stuff

        # Params as theano.shared matrices
        # W for x, V for Lxm, U for h (small), S for h (big), R for hm, 
        self.L = shared(random_weight_matrix(wdim, vdim), name='L')
        self.W = shared(random_weight_matrix(hdim, wdim), name='W')
        self.V = shared(random_weight_matrix(hdim, wdim), name='V')
        self.U = shared(random_weight_matrix(hdim, hdim), name='U')
        self.S = shared(random_weight_matrix(hdim, hdim), name='S')
        self.R = shared(random_weight_matrix(hdim, hdim), name='R')

        self.params = [self.L, self.W, self.V, self.U, self.S, self.R]
        self.vparams = [0.0*param.get_value() for param in self.params]

        # # Compile stuff
        # xs = T.imatrix('xs')
        # self.f = function([xs], self.symbolic_f_prop(xs), allow_input_downcast=True)

    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0 * dparam.get_value())

    def rnn_timestep(self, x_t, h_prev, Lxm_t):
        # So simple!
        Lx_t = self.L[:,x_t]
        h_t = T.tanh(T.dot(self.W, Lx_t) + T.dot(self.V, Lxm_t) + T.dot(self.U, h_prev))
        return h_t
        
    def meta_rnn_timestep(self, xm_t, h_meta_prev, num_examples, xs):
        
        Lxm_t = self.L[:,xm_t]
        
        # calculate regular rnn timestep
        h_prev = T.zeros([self.hdim, num_examples], dtype='float64')
        results, updates = scan(fn = self.rnn_timestep, 
                                outputs_info = h_prev,
                                sequences = xs,
                                non_sequences = Lxm_t)
        h_t = results[-1]

        h_meta_t = T.tanh(T.dot(self.S, h_t) + T.dot(self.R, h_meta_prev))
        return h_meta_t


    def reg_updates_cost(self):
        param_values = [param.get_value() for param in self.params]
        updates = [self.rho * param for param in param_values]
        reg_cost = 0.5 * self.rho * (np.sum(np.sum(param**2) for param in param_values))
        return (updates, reg_cost)

    def symbolic_f_prop(self, xs):
        """returns symbolic variable based on ch_prev and xs."""
        # assert(len(xs[0]) > 0)
        num_examples = xs.shape[1]
        # print xs.type
        h_meta_prev = T.zeros([self.hdim, num_examples], dtype='float64')
        results, updates = scan(fn = self.meta_rnn_timestep, 
                                outputs_info = h_meta_prev,
                                sequences = xs,
                                non_sequences = [num_examples, xs])
        return results[-1]


    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams

if __name__ == '__main__':
    print 'Sanity check'
    new = NewEnc(15,15,15)
    xs = np.array([1,2,3]).reshape([-1,1])
    print new.f(xs)
    #ch = gru.f_prop(xs)
    #print ch
    
