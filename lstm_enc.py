import numpy as np
import theano.tensor as T
from theano import function, shared, scan
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
        # self.Uy = shared(random_weight_matrix(vdim, hdim), name='Uy')
        # skipping biases (at least for now) for simplicity
        
        # # Inputs to calculate next h
        # x_t = T.scalar('x_t', dtype='int32') # TODO: what's the best type of int?
        # # y_t = T.scalar('y_t', dtype='int32')
        # ch_prev = T.vector('ch_prev')
        # c_prev = ch_prev[:self.hdim]#T.vector('c_prev')
        # h_prev = ch_prev[self.hdim:]#T.vector('h_prev')

        # Lx_t = self.L[:,x_t]


        # # gates (input, forget, output)
        # i_t = sigmoid(T.dot(self.Wi, Lx_t) + T.dot(self.Ui, h_prev))
        # f_t = sigmoid(T.dot(self.Wf, Lx_t) + T.dot(self.Uf, h_prev))
        # o_t = sigmoid(T.dot(self.Wo, Lx_t) + T.dot(self.Uo, h_prev))
        # # new memory cell
        # c_new_t = T.tanh(T.dot(self.Wc, Lx_t) + T.dot(self.Uc, h_prev))
        # # final memory cell
        # c_t = f_t * c_prev + i_t * c_new_t
        # # final hidden state
        # h_t = o_t * T.tanh(c_t)
        # # final joint state
        # ch_t = T.concatenate([h_t, c_t])
        # # Compile!
        # self.calc_ch = function([x_t, ch_prev], ch_t)

        # Anything else? maybe...
        self.f_prop_function = self.compile_f_prop()

    def lstm_timestep(self, x_t, ch_prev):
        """calculates info to pass to next time step.
        x_t is a scalar; ch_prev is a vector of size 2*hdim"""

        c_prev = ch_prev[:self.hdim]#T.vector('c_prev')
        h_prev = ch_prev[self.hdim:]#T.vector('h_prev')

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
        # final joint state
        ch_t = T.concatenate([c_t, h_t])

        return ch_t

    def symbolic_f_prop(self, xs, ch_prev):
        """returns symbolic variable based on ch_prev and xs."""

        results, updates = scan(fn = self.lstm_timestep, 
                                outputs_info = ch_prev,
                                sequences = xs)
        return results[-1]


    def compile_f_prop(self):
        """one-time create f_prop function"""
        ch_prev = T.vector('ch_prev') # through dimensions at one time
        xs = T.ivector('xs') # through time

        return function([xs, ch_prev], self.symbolic_f_prop(xs, ch_prev))
        

    def compile_b_prop(self):
        """one-time create b_prop function"""
        delta = T.vector('delta')
        # delta represents derivative of cost wrt ch_inter
        # gradient of ch_inter wrt each param
        # TODO: figure out tomorrow...
        pass


    def f_prop(self, xs):
        return self.f_prop_function(xs,np.zeros(2*self.hdim))


if __name__ == '__main__':
    print 'Sanity check'
    le = LSTMEnc(15,15,15)
    xs = [1,2,3]
    ch = le.f_prop(xs)
    print ch
