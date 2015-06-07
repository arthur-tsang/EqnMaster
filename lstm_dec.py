import numpy as np
import theano.tensor as T
from theano import function, shared, scan, pp, scan_module
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
        self.out_end = outdim # the end token

        # Parameters
        np.random.seed(rseed)

        # Learning rate
        self.alpha = alpha

        # Regularization
        self.rho = rho


        ## Theano stuff

        # Params as theano.shared matrices
        # W: times character-vector, U: times previous-hidden-vector
        # i: input, f: forget, o: output, c: new-cell
        self.Ui = shared(random_weight_matrix(hdim, hdim), name='Ui')
        self.Uf = shared(random_weight_matrix(hdim, hdim), name='Uf')
        self.Uo = shared(random_weight_matrix(hdim, hdim), name='Uo')
        self.Uc = shared(random_weight_matrix(hdim, hdim), name='Uc')
        self.U  = shared(random_weight_matrix(outdim, hdim), name='U')
        self.b  = shared(np.zeros([outdim, 1]), name='b', broadcastable=(False, True))

        self.params = [self.Ui, self.Uf, self.Uo, self.Uc, self.U, self.b]
        self.vparams = [0.0*param.get_value() for param in self.params]

        # # symbolic generate
        # ch_prev = T.vector('ch_prev')
        # self.generate_function = function([ch_prev], self.symbolic_generate(ch_prev))
        # print 'done compiling'

    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0.0 * dparam.get_value())

    def lstm_timestep(self, y_t, old_cost, ch_prev):
        """calculates info to pass to next time step.
        ch_prev is a vector of size 2*hdim"""

        y_filtered_ind = T.ge(y_t, 0).nonzero()[0]
        y_filtered = y_t[y_filtered_ind]

        # break up into c and h
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
        y_hat_t = softmax(theta_t.T).T
        # Softmax wraps output in another list, why??
        # (specifically it outputs a 2-d row, not a 1-d column)
        # y_hat_t = y_hat_t[0]
        # Compute new cost # TODO
        cost = T.sum(-T.log(y_hat_t[y_filtered, y_filtered_ind]))

        new_cost = old_cost + cost

        # final joint state
        ch_t = T.concatenate([c_t, h_t])

        return new_cost, ch_t

    def reg_updates_cost(self):
        """returns list of param updates and cost due to regularization"""
        param_values = [param.get_value() for param in self.params]
        updates = [self.rho * param if len(param.shape) > 1 else 0 * param for param in param_values]
        reg_cost = 0.5 * self.rho * (np.sum(np.sum(param**2) for param in param_values if len(param.shape) > 1))
        return (updates, reg_cost)


    def symbolic_f_prop(self, ys, h_prev):
        """returns symbolic variable based on ys and h_prev."""

        # Make sure all the examples in ys have the same length by this point
        # (by padding with -1)
        # ys = np.array(ys)

        results, updates = scan(fn = self.lstm_timestep, 
                                outputs_info = [np.float64(0.0), h_prev],
                                sequences=ys)

        # Return the cost (index 0) at the most recent timestep (-1)
        return results[0][-1]


    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams

    def lstm_output(self, y_prev, ch_prev):
        """calculates info to pass to next time step.
        ch_prev is a vector of size 2*hdim"""

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
        y_hat_t = softmax(theta_t.T).T
        # Softmax wraps output in another list, why??
        # (specifically it outputs a 2-d row, not a 1-d column)
        # y_hat_t = y_hat_t[0]
        # Compute new cost
        out_label = T.argmax(y_hat_t)

        # final joint state
        ch_t = T.concatenate([c_t, h_t])

        return (out_label, ch_t), scan_module.until(T.eq(out_label, self.out_end))

    def symbolic_generate(self, h_prev):
        """generate ys from a given h_prev"""

        results, updates = scan(fn = self.lstm_output, 
                                outputs_info = [np.int64(0), h_prev],
                                n_steps = 50)


        return results[0]
        
    # def generate(self, ch_prev):
    #     return self.generate_function(ch_prev)



if __name__ == '__main__':
    print 'Sanity check'
    ld = LSTMDec(10,10,10)
    # ys = [1,2,3,4]
    # ch_prev = np.ones(2*ld.hdim)
    # cost_final = ld.f_prop(ys, ch_prev)
    # print cost_final
    
    # ld.b_prop(ys, ch_prev)
    # print 'printing dparams'
    # for dparam in ld.dparams:
    #     print dparam.get_value()

    print 'testing generate'
    print ld.generate(np.ones(2*ld.hdim))
    print 'done testing'

    
