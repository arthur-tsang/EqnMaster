import numpy as np
import theano.tensor as T
from theano import function, shared, scan, pp
from theano.tensor.nnet import sigmoid, softmax

from misc import random_weight_matrix

# # For debugging:
# from theano import config
# config.exception_verbosity = 'high'

rng = np.random


class LSTMDec:
    
    def __init__(self, hdim, outdim, alpha=.005, rho=.0001, rseed=10):
        
        # Dimensions
        self.hdim = hdim
        self.outdim = outdim

        # Parameters
        np.random.seed(rseed)

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho


        ## Theano stuff

        # Params as theano.shared matrices
        # self.L = shared(random_weight_matrix(wdim, vdim), name='L')
        # W: times character-vector, U: times previous-hidden-vector
        # i: input, f: forget, o: output, c: new-cell
        # self.Wi = shared(random_weight_matrix(hdim, wdim), name='Wi')
        self.Ui = shared(random_weight_matrix(hdim, hdim), name='Ui')
        # self.Wf = shared(random_weight_matrix(hdim, wdim), name='Wf')
        self.Uf = shared(random_weight_matrix(hdim, hdim), name='Uf')
        # self.Wo = shared(random_weight_matrix(hdim, wdim), name='Wo')
        self.Uo = shared(random_weight_matrix(hdim, hdim), name='Uo')
        # self.Wc = shared(random_weight_matrix(hdim, wdim), name='Wc')
        self.Uc = shared(random_weight_matrix(hdim, hdim), name='Uc')
        self.U  = shared(random_weight_matrix(outdim, hdim), name='U')
        self.b  = shared(np.zeros(outdim), name='b')

        # # self.dL = shared(np.zeros(wdim, vdim), name='dL')
        # # W: times character-vector, U: times previous-hidden-vector
        # # i: input, f: forget, o: output, c: new-cell
        # # self.dWi = shared(np.zeros(hdim, wdim), name='dWi')
        # self.dUi = shared(np.zeros((hdim, hdim)), name='dUi')
        # # self.dWf = shared(np.zeros(hdim, wdim), name='dWf')
        # self.dUf = shared(np.zeros((hdim, hdim)), name='dUf')
        # # self.dWo = shared(np.zeros(hdim, wdim), name='dWo')
        # self.dUo = shared(np.zeros((hdim, hdim)), name='dUo')
        # # self.dWc = shared(np.zeros(hdim, wdim), name='dWc')
        # self.dUc = shared(np.zeros((hdim, hdim)), name='dUc')
        # self.dU  = shared(np.zeros((hdim, hdim)), name='dU')
        # self.db  = shared(np.zeros(outdim), name='db')

        self.params = [self.Ui, self.Uf, self.Uo, self.Uc, self.U, self.b]

        # # self.params = [self.L, self.Wi, self.Ui, self.Wf, self.Uf, self.Wo, self.Uo, self.Wc, self.Uc, self.U, self.db]
        # self.dparams = [self.dUi, self.dUf, self.dUo, self.dUc, self.dU, self.db]
        # # self.dparams = [self.dL, self.dWi, self.dUi, self.dWf, self.dUf, self.dWo, self.dUo, self.dWc, self.dUc, self.dU, self.db]

        # # compile functions!
        # self.f_prop_function = self.compile_f_prop() 
        # self.b_prop_function = self.compile_b_prop()
        # print 'done compiling functions'


    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0.0 * dparam.get_value())


    def lstm_timestep(self, y_t, old_cost, ch_prev):
        """calculates info to pass to next time step.
        x_t is a scalar; ch_prev is a vector of size 2*hdim"""

        c_prev = ch_prev[:self.hdim]#T.vector('c_prev')
        h_prev = ch_prev[self.hdim:]#T.vector('h_prev')

        # gates (input, forget, output)
        i_t = sigmoid(T.dot(self.Ui, h_prev))
        f_t = sigmoid(T.dot(self.Uf, h_prev))
        o_t = sigmoid(T.dot(self.Uo, h_prev))
        # new memory cell
        c_new_t = T.tanh(T.dot(self.Uc, h_prev))
        # final memory cell
        c_t = f_t * c_prev + i_t * c_new_t
        # final hidden state
        h_t = o_t * T.tanh(c_t)

        # Input vector for softmax
        theta_t = T.dot(self.U, h_t) + self.b
        # Softmax prob vector
        y_hat_t = softmax(theta_t)
        # Softmax wraps output in another list, why??
        y_hat_t = y_hat_t[0]
        # Compute new cost
        cost = -T.log(y_hat_t[y_t])

        new_cost = old_cost + cost

        # final joint state
        ch_t = T.concatenate([c_t, h_t])

        return new_cost, ch_t

    def symbolic_f_prop(self, ys, ch_prev):
        """returns symbolic variable based on ys and ch_prev."""

        results, updates = scan(fn = self.lstm_timestep, 
                                outputs_info = [np.float64(0.0), ch_prev],
                                sequences=ys)

        # Results is a matrix!!
        return results[0][-1]


    # def compile_f_prop(self):
    #     """one-time create f_prop function"""
    #     ch_prev = T.vector('ch_prev') # through dimensions at one time
    #     ys = T.ivector('ys') # through time

    #     return function([ys, ch_prev], self.symbolic_f_prop(ys, ch_prev))


        
    # def f_prop(self, ys, ch_prev):
    #     final_cost = self.f_prop_function(ys, ch_prev)
    #     return final_cost


    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams
        
    # def compile_b_prop(self):
    #     # cost_final is symbolic (output of symbolic_f_prop)
    #     # TODO: would the function be faster if it took in hs?
    #     # TODO: also, make sure this doesn't take too long (on order 10s right now for me)
    #     ch_prev = T.vector('ch_prev')
    #     ys = T.ivector('ys')
    #     cost_final = self.symbolic_f_prop(ys, ch_prev)

    #     # print 'cost final', pp(cost_final), cost_final.type

    #     print 'working on compiling backprop'
    #     return function([ys, ch_prev], self.symbolic_b_prop(cost_final))

    # def b_prop(self, ys, ch_prev):
    #     new_dparams = self.b_prop_function(ys, ch_prev)
    #     for dparam, new_dparam in zip(self.dparams, new_dparams):
    #         dparam.set_value(new_dparam + dparam.get_value())


    # # new approach
    # def b_prop(self, cost_final_numeric):
    #     new_dparams = []
    #     for param in self.params:
    #         new_dparams.append(T.grad(cost_final, param))
        


    # TODO: write a decode_sequence function


if __name__ == '__main__':
    print 'Sanity check'
    ld = LSTMDec(10,10,10)
    ys = [1,2,3,4]
    ch_prev = np.ones(2*ld.hdim)
    cost_final = ld.f_prop(ys, ch_prev)
    print cost_final
    
    ld.b_prop(ys, ch_prev)
    print 'printing dparams'
    for dparam in ld.dparams:
        print dparam.get_value()

    #self.b_prop_function

# The following might be related to my gradient problem right now
# http://tiku.io/questions/2870308/defining-a-gradient-with-respect-to-a-subtensor-in-theano
