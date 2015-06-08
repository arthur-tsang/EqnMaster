import numpy as np
import theano.tensor as T
import pickle
from theano import function, shared, scan
from theano.tensor.nnet import sigmoid, softmax

from misc import random_weight_matrix
import signal

rng = np.random

class DNN:

    def __init__(self, vdim, hdim, wdim, outdim=2, alpha=.005, rho=.0001, mu=0.75, rseed=10):
        
        # Dimensions
        self.vdim = vdim
        assert(wdim == hdim) # so it follows the forn of everything else
        #self.hdim = hdim 
        self.wdim = wdim
        self.outdim = outdim

        # Parameters
        np.random.seed(rseed)

        # Learning rate
        self.alpha = alpha
        self.mu = mu
        self.rho = rho
        self.rseed = rseed


        ## Theano stuff

        # Params as theano.shared matrices
        self.L = shared(random_weight_matrix(wdim, vdim), name='L')
        # W: times character-vector, U: times previous-hidden-vector
        #self.Wx = shared(random_weight_matrix(hdim, wdim), name='Wx')
        #self.Wh = shared(random_weight_matrix(wdim, wdim), name='Wh')
        self.U = shared(random_weight_matrix(hdim, wdim), name='U')
        self.b  = shared(np.zeros([hdim, 1]), name='b', broadcastable=(False, True))
        self.U2 = shared(random_weight_matrix(outdim, hdim), name='U2') # for the second layer
        self.b2  = shared(np.zeros([outdim, 1]), name='b2', broadcastable=(False, True))


        self.params = [self.L, self.U, self.b, self.U2, self.b2]
        self.vparams = [0.0*param.get_value() for param in self.params]

        self.prop_compiled = self.compile_self()
        self.generate_compiled = self.compile_generate()

    def reset_grads(self):
        """Resets all grads to zero (maintaining shape!)"""
        for dparam in self.dparams:
            dparam.set_value(0 * dparam.get_value())

    def dnn_timestep(self, x_t, old_cost, h_prev, ys):
        """Basically, part of a summation of Lx's"""
        # h_prev is of size wdim
        Lx_t = self.L[:,x_t]
        # gates (update, reset)
        # h_t = T.tanh(T.dot(self.Wx, Lx_t) + T.dot(self.Wh, h_prev))
        h_t = h_prev + Lx_t # just a stupid linear combination
        first_layer = T.tanh(T.dot(self.U, h_t) + self.b)
        y_hat_t = softmax((T.dot(self.U2, first_layer) + self.b2).T).T
        cost = T.sum(-T.log(y_hat_t[ys, T.arange(ys.shape[0])]))
        return cost, h_t

    def reg_updates_cost(self):
        param_values = [param.get_value() for param in self.params]
        updates = [self.rho * param for param in param_values]
        reg_cost = 0.5 * self.rho * (np.sum(np.sum(param**2) for param in param_values))
        return (updates, reg_cost)

    def symbolic_f_prop(self, xs, ys):
        """returns symbolic variable based on ch_prev and xs."""
        num_examples = xs.shape[1]
        h_prev = T.zeros([self.wdim, num_examples], dtype='float64')
        results, updates = scan(fn = self.dnn_timestep, 
                                outputs_info = [np.float64(0), h_prev],
                                sequences = xs,
                                non_sequences = ys)
        return results[0][-1]


    def symbolic_b_prop(self, cost_final):
        new_dparams = []
        for param in self.params:
            new_dparams.append(T.grad(cost_final, param))

        return new_dparams


    def compile_self(self):
        """Compiles a function that computes both cost and deltas at the same time"""

        # every row of xs is a timestep, each column an example; likewise for ys (must be padded)
        xs = T.imatrix('xs')
        ys = T.ivector('ys')
            
        cost = self.symbolic_f_prop(xs, ys)
        new_dparams = self.symbolic_b_prop(cost)

        return function([xs, ys], [cost] + new_dparams, allow_input_downcast=True)


    def prop(self, xs, ys):

        """Like f_prop, but also returns updates for bprop"""
        # return self.both_prop_compiled(xs, ys + [self.out_end])
        return self.prop_compiled(xs, ys)


    def dnn_output(self, x_t, old_label, h_prev):

        Lx_t = self.L[:,x_t]
        h_t = Lx_t + h_prev
        #h_t = T.tanh(T.dot(self.Wx, Lx_t) + T.dot(self.Wh, h_prev))
        print h_t.type
        first_layer = T.tanh(T.dot(self.U, h_t) + self.b.reshape([-1]))
        y_hat_t = softmax((T.dot(self.U2, first_layer) + self.b2.reshape([-1])).T).T
        #y_hat_t = softmax(T.dot(self.U, h_t) + self.b)[0]


        out_label = T.argmax(y_hat_t) # good line
        #out_label = first_layer.shape[1] # bad line

        return out_label, h_t


    def symbolic_output(self, xs):
        """generate ys from a given h_prev"""
        h_prev = T.zeros(self.wdim, dtype='float64')
        results, updates = scan(fn = self.dnn_output, 
                                outputs_info = [np.int64(0), h_prev],
                                sequences=xs)


        return results[0][-1]

    def compile_generate(self):
        xs = T.ivector('xs')
        return function([xs], self.symbolic_output(xs), allow_input_downcast=True)

    def generate_answer(self, xs):
        xs = np.array(xs)
        ys = self.generate_compiled(xs)
        return ys

    def update_params(self, new_dparams, update_rule):
        """Updates params according to deltas given"""

        num_params = len(self.params)

        # Model update
        for index, dparam in enumerate(new_dparams[:num_params]):
            if update_rule == 'sgd':
                self.params[index].set_value(self.params[index].get_value() - self.alpha * dparam)
            elif update_rule == 'momentum':
                v_prev = self.vparams[index]
                v = v_prev*self.mu  - self.alpha * dparam

                self.params[index].set_value(self.params[index].get_value() + v)
                self.vparams[index] = v


    def process_batch(self, all_xs, all_ys, shouldUpdate = True, update_rule='momentum'):
        """Worry about end token and padding to use this function!
        Also this function does all the updates for you"""
        assert(len(all_xs) > 0)
        batch_size = all_xs.shape[1]
        cost_and_dparams = self.prop(all_xs, all_ys)
        tot_cost = cost_and_dparams[0]
        dparams = [dparam/float(batch_size) for dparam in cost_and_dparams[1:]]
    

        # Regularization
        reg_updates, reg_cost = self.reg_updates_cost()

        dparams_tot = [(avg + reg) for (avg, reg) in zip(dparams, reg_updates)]

        if shouldUpdate:
            self.update_params(dparams_tot, update_rule)
            # self.update_params(d_reg_updates + e_reg_updates)
        
        final_cost = (float(tot_cost) / batch_size) + reg_cost
        
        return final_cost


    def sgd(self, batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True, update_rule='momentum', filename='models/tmp.p'):
        """Implentation of minibatch SGD over all training data (copied from enc_dec). End-tokens will be automatically added later"""
        best_dev_cost = 1000000.0
        # partitions is a list of 2D lists, one for each input length
        partitionX, partitionY = self.partition_XY(X_train, Y_train)
        # Weights for random sampling, given by how many examples are there for each size
        size_probs = np.array([len(size_matrix) for size_matrix in partitionX])
        size_probs = size_probs/float(np.sum(size_probs))
        print "Sum of Probs:", np.sum(size_probs)
        print 'Train Set Size:', len(Y_train)

        N = len(X_train)
        iterations_per_epoch = 1+ N / batch_size # using SGD

        # 1 epoch is 1 pass over training data
        for epoch in xrange(n_epochs):
            print "Epoch:", epoch
            # For every sub-iteration
            for i in xrange(iterations_per_epoch):
                # print i
                # Sample a size
                rand_size_index = np.random.choice(len(partitionX), p=size_probs)
                curr_subset_X = np.array(partitionX[rand_size_index])
                curr_subset_Y = np.array(partitionY[rand_size_index])

                curr_batch_size = min(batch_size, len(curr_subset_X))

                batch_mask = np.random.choice(len(curr_subset_X), curr_batch_size, replace=False)
                X_batch = curr_subset_X[batch_mask]
                X_batch = np.transpose(X_batch)

                Y_batch = curr_subset_Y[batch_mask]
                Y_batch = np.transpose(Y_batch)

                avg_cost = self.process_batch(X_batch, Y_batch, shouldUpdate = True, update_rule=update_rule) # takes care of sgd

            # Print progress
            if verbose and (epoch % 10) == 0:
                print "Epoch", epoch
                tot_cost = 0.0
                for i in range(min(N,50)):
                    single_X = np.array(X_train[i]).reshape([-1, 1])
                    single_Y = np.array([Y_train[i]])
                    tot_cost += self.process_batch(single_X, single_Y, shouldUpdate = False)
                print "Training Cost (estimate):", tot_cost/50.0
                if X_dev is not None:
                    tot_cost = 0.0
                    for i in range(min(N,50)):
                        single_X = np.array(X_dev[i]).reshape([-1, 1])
                        single_Y = np.array([Y_dev[i]])
                        tot_cost += self.process_batch(single_X, single_Y, shouldUpdate = False)
                    print "Dev Cost:", tot_cost/50.0
                    if tot_cost/50.0 <= best_dev_cost:
                        print "Saving"
                        self.save_model(filename)
                        best_dev_cost = tot_cost/50.0


    def partition_XY(self, X_train, Y_train):
        partitionX = {}
        partitionY = {}
        for x_list, y_list in zip(X_train, Y_train):
            curr_len = len(x_list)
            if curr_len in partitionX:
                partitionX[curr_len].append(x_list)
                partitionY[curr_len].append(y_list)
            else:
                partitionX[curr_len] = [x_list]
                partitionY[curr_len] = [y_list]

        final_partitionX = [partitionX[key] for key in partitionX]
        final_partitionY = [partitionY[key] for key in partitionY]
        return final_partitionX, final_partitionY


    def save_model(self, file_name):
        # Save encoder/decoder to a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_model(self, file_name):
        # Load encoder/decoder from a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            for self_param, saved_param in zip(self.params, params):
                self_param.set_value(saved_param.get_value())


if __name__ == '__main__':
    dnn= DNN(12,12,12)
    X_train = [[1,2,3], [4,5], [1,2,4]]
    Y_train = [1, 0, 0]
    dnn.sgd(5, 1000, X_train, Y_train)
    print dnn.generate_answer([1,2,3])
    print dnn.generate_answer([4,5])
    print dnn.generate_answer([1,2,4])
    # print 'Sanity check'
    # gru = GRUEnc(15,15,15)
    # xs = [1,2,3]
    #ch = gru.f_prop(xs)
    #print ch
    
