import numpy as np
import theano.tensor as T
from theano import function, shared
from theano.tensor.nnet import sigmoid, softmax

from misc import random_weight_matrix

rng = np.random

class LSTMEnc:

    def __init__(self, vdim, hdim, wdim, alpha=.005, rho=.0001, rseed=10):
        
        # Dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.vdim = vdim
        self.wdim = wdim

        # Parameters
        np.random.seed(rseed)
        sigma = .1

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho


        ## Theano stuff

        # Params as theano.shared matrices
        self.L = shared(random_weight_matrix(wdim, vdim), name='L')
        # W: times character-vector, U: times previous-hidden-vector
        # i: input, f: forget, o: output, c: new-cell
        self.Wi = shared(random_weight_matrix(hdim, wdim), name='Wi')
        self.Ui = shared(random_weight_matrix(hdim, hdim), name='Ui')
        self.Wf = shared(random_weight_matrix(hdim, wdim), name='Wf')
        self.Uf = shared(random_weight_matrix(hdim, hdim), name='Uf')
        self.Wo = shared(random_weight_matrix(hdim, wdim), name='Wo')
        self.Uo = shared(random_weight_matrix(hdim, hdim), name='Uo')
        self.Wc = shared(random_weight_matrix(hdim, wdim), name='Wc')
        self.Uc = shared(random_weight_matrix(hdim, hdim), name='Uc')
        # and for computing yhat and costs
        self.Uy = shared(random_weight_matrix(vdim, hdim), name='Uy')
        # skipping biases (at least for now) for simplicity
        
        # Inputs to calculate next h
        x_t = T.scalar('x_t', dtype='int32') # TODO: what's the best type of int?
        y_t = T.scalar('y_t', dtype='int32')
        h_prev = T.vector('h_prev')
        c_prev = T.vector('c_prev')
        Lx_t = self.L[:,x_t]


        # gates (input, forget, output)
        i_t = sigmoid(T.dot(self.Wi, Lx_t) + T.dot(self.Ui, h_prev))
        f_t = sigmoid(T.dot(self.Wf, Lx_t) + T.dot(self.Uf, h_prev))
        o_t = sigmoid(T.dot(self.Wo, Lx_t) + T.dot(self.Uo, h_prev))
        # new memory cell
        c_new_t = T.tanh(T.dot(self.Wc, Lx_t) + T.dot(self.Uc, h_prev))
        # final memory cell
        c_t = f_t * c_prev + i_t * c_new_t
        # final hidden state
        h_t = o_t * T.tanh(c_t)
        # cost
        y_hats = softmax(T.dot(self.Uy, h_t))
        cell_cost = -T.log(y_hats[y_t])

        # Compile!
        self.calc_h = function([x_t, h_prev, c_prev], h_t)
        self.calc_cost = function([h_t, y_t], cell_cost)

        # Anything else? maybe...

    def f_prop(self, xs, ys):
        x = xs[0]
        y = ys[0]
        print self.calc_h(x, np.zeros(5), np.zeros(5))
        # TODO: figure out how to use scan to do f_prop efficiently!


if __name__ == '__main__':
    print 'Sanity check'
    le = LSTMEnc(5,5,5)
    xs = np.array([1,2,3])
    ys = np.array([2,3,4])
    le.f_prop(xs, ys)
