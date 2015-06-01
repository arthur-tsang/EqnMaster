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

        self.dL = shared(np.zeros(wdim, vdim), name='dL')
        # W: times character-vector, U: times previous-hidden-vector
        # i: input, f: forget, o: output, c: new-cell
        self.dWi = shared(np.zeros(hdim, wdim), name='dWi')
        self.dUi = shared(np.zeros(hdim, hdim), name='dUi')
        self.dWf = shared(np.zeros(hdim, wdim), name='dWf')
        self.dUf = shared(np.zeros(hdim, hdim), name='dUf')
        self.dWo = shared(np.zeros(hdim, wdim), name='dWo')
        self.dUo = shared(np.zeros(hdim, hdim), name='dUo')
        self.dWc = shared(np.zeros(hdim, wdim), name='dWc')
        self.dUc = shared(np.zeros(hdim, hdim), name='dUc')

        self.params = [self.L, self.Wi, self.Ui, self.Wf, self.Uf, self.Wo, self.Uo, self.Wc, self.Uc]
        self.dparams = [self.dL, self.dWi, self.dUi, self.dWf, self.dUf, self.dWo, self.dUo, self.dWc, self.dUc]

        self.f_prop_function = self.compile_f_prop() 
        self.b_prop_function = self.compile_b_prop()

    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0 * dparam.get_value())

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
        
    def f_prop(self, xs):
        return self.f_prop_function(xs,np.zeros(2*self.hdim))

    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams
        
    def compile_b_prop(self):
        cost_final = T.scalar('cost_final')
        return function([cost_final], self.symbolic_b_prop(cost_final))

    def b_prop(self, cost_final):
        new_dparams = self.b_prop_function(cost_final)
        for dparam, new_dparam in zip(self.dparams, new_dparams):
            dparam.set_value(new_dparam + dparam.get_value())
        


if __name__ == '__main__':
    print 'Sanity check'
    le = LSTMEnc(15,15,15)
    xs = [1,2,3]
    ch = le.f_prop(xs)
    print ch
