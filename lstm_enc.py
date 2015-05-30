import numpy as np
import theano.tensor as T
from theano import function, shared

from misc import random_weight_matrix

sigmoid = T.nnet.sigmoid
rng = numpy.random

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

        # self.params = {}
        # self.params['L'] = np.random.normal(0, sigma, (wdim, vdim)) # "wide" array
        # # Input gate params
        # self.params['Wi'] = random_weight_matrix(hdim, wdim)
        # self.params['Ui'] = random_weight_matrix(hdim, hdim)
        # # Forget gate params
        # self.params['Wf'] = random_weight_matrix(hdim, wdim)
        # self.params['Uf'] = random_weight_matrix(hdim, hdim)
        # # Output gate params
        # self.params['Wo'] = random_weight_matrix(hdim, wdim)
        # self.params['Uo'] = random_weight_matrix(hdim, hdim)
        # # "New" memory cell params
        # self.params['Wc'] = random_weight_matrix(hdim, wdim)
        # self.params['Uc'] = random_weight_matrix(hdim, hdim)
        
        # skipping biases (at least for now) for simplicity

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho


        # start doing theano stuff
        # define complicated function
        # store that in self.something...
        # Wi = T.dmatrix('Wi')
        # Ui = T.dmatrix('Ui')
        # Wf = T.dmatrix('Wf')
        # Uf = T.dmatrix('Uf')
        # Wo = T.dmatrix('Wo')
        # Uo = T.dmatrix('Uo')
        # Wc = T.dmatrix('Wc')
        # Uc = T.dmatrix('Uc')
        # params as theano.shared matrices
        self.Wi = shared(random_weight_matrix(hdim, wdim), name='Wi')
        self.Ui = shared(random_weight_matrix(hdim, hdim), name='Ui')
        self.Wf = shared(random_weight_matrix(hdim, wdim), name='Wf')
        self.Uf = shared(random_weight_matrix(hdim, hdim), name='Uf')
        self.Wo = shared(random_weight_matrix(hdim, wdim), name='Wo')
        self.Uo = shared(random_weight_matrix(hdim, hdim), name='Uo')
        self.Wc = shared(random_weight_matrix(hdim, wdim), name='Wc')
        self.Uc = shared(random_weight_matrix(hdim, hdim), name='Uc')
        
        
        x_t = T.vector('x_t')
        h_prev = T.vector('h_prev')

        # gates (input, forget, output)
        i_t = sigmoid(T.dot(self.Wi, x_t) + T.dot(self.Ui, h_prev))
        f_t = sigmoid(T.dot(self.Wf, x_t) + T.dot(self.Uf, h_prev))
        o_t = sigmoid(T.dot(self.Wo, x_t) + T.dot(self.Uo, h_prev))
        # new memory cell
        c_new_t = T.tanh(T.dot(self.Wc, x_t) + T.dot(self.Uc, h_prev))
        # final memory cell
        c_t = f_t * c_prev + i_t * c_new_t
        # final hidden state
        h_t = o_t * T.tanh(c_t)
        
        # Compile!
        self.next_h = function([x_t, h_prev], h_t)
        

        # Anything else? maybe...

    def f_prop(self, xs, ys):
        x = xs[0]
        y = ys[0]
        print self.next_h(x, np.zeros(5))


if __name__ == '__main__':
    print 'Sanity check'
    le = LSTMEnc(5,5,5)
    xs = np.array([1,2,3])
    ys = np.array([2,3,4])
    le.f_prop(xs, ys)
