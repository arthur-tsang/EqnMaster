import numpy as np
import theano.tensor as T
from theano import function, shared, scan
from theano.tensor.nnet import sigmoid, softmax

from misc import random_weight_matrix

rng = np.random

class GRUEnc:

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
        self.L = shared(random_weight_matrix(wdim, vdim), name='L')
        # W: times character-vector, U: times previous-hidden-vector
        # z: update, r: reset, h: new memory content (my notation)
        self.Wz = shared(random_weight_matrix(hdim, wdim), name='Wz')
        self.Uz = shared(random_weight_matrix(hdim, hdim), name='Uz')
        self.Wr = shared(random_weight_matrix(hdim, wdim), name='Wr')
        self.Ur = shared(random_weight_matrix(hdim, hdim), name='Ur')
        self.Wh = shared(random_weight_matrix(hdim, wdim), name='Wh')
        self.Uh = shared(random_weight_matrix(hdim, hdim), name='Uh')

        self.params = [self.L, self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]
        self.vparams = [0.0*param.get_value() for param in self.params]

    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0 * dparam.get_value())

    def gru_timestep(self, x_t, h_prev):

        Lx_t = self.L[:,x_t]
        # gates (update, reset)
        z_t = sigmoid(T.dot(self.Wz, Lx_t) + T.dot(self.Uz, h_prev))
        r_t = sigmoid(T.dot(self.Wr, Lx_t) + T.dot(self.Ur, h_prev))
        # combine them
        h_new_t = T.tanh(T.dot(self.Wh, Lx_t) + r_t * T.dot(self.Uh, h_prev))
        h_t = z_t * h_prev + (1 - z_t) * h_new_t
        return h_t

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
        h_prev = T.zeros([self.hdim, num_examples], dtype='float64')
        # print type(h_prev[0, 0])
        results, updates = scan(fn = self.gru_timestep, 
                                outputs_info = h_prev,
                                sequences = xs)
        return results[-1]


    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams

if __name__ == '__main__':
    print 'Sanity check'
    gru = GRUEnc(15,15,15)
    xs = [1,2,3]
    #ch = gru.f_prop(xs)
    #print ch
    
