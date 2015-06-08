import numpy as np
import pickle
import os.path
from itertools import takewhile

import theano.tensor as T
from theano import function

from gru_enc import GRUEnc
from gru_dec import GRUDec

# # For debugging:
from theano import config
config.floatX = 'float32'
# config.exception_verbosity='high'
# config.optimizer = 'fast_compile'
# config.exception_verbosity = 'high'


class NewEncDec:
    def __init__(self, vdim, hdim, wdim, outdim, alpha=.005, rho=.0001, mu=0.75, rseed = 10):
        # dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.wdim = wdim
        self.outdim = outdim + 1
        self.out_end = self.outdim - 1 # idx of end token

        # others (I don't think we'll need these, save for saving/loading)
        self.alpha = alpha
        self.mu = mu
        self.rho = rho
        self.rseed = rseed

        # sub-models
        self.encoder = GRUEnc(self.vdim, self.hdim, self.wdim, alpha=alpha, rho=rho, rseed=rseed)
        self.decoder = GRUDec(self.hdim, self.outdim, alpha=alpha, rho=rho, rseed=rseed)
        
        # compiled functions
        print 'about to compile'
        self.both_prop_compiled = self.compile_both()
        self.generate_function = self.compile_generate()
        print 'done compiling'

    def symbolic_f_prop(self, xs, ys):
        hidden_inter = self.encoder.symbolic_f_prop(xs)
        cost = self.decoder.symbolic_f_prop(ys, hidden_inter)
        return cost

    def symbolic_b_prop(self, cost):
        dec_new_dparams = self.decoder.symbolic_b_prop(cost)
        enc_new_dparams = self.encoder.symbolic_b_prop(cost)

        return dec_new_dparams + enc_new_dparams # python-list concatenation

    def compile_both(self):
        """Compiles a function that computes both cost and deltas at the same time"""

        # every row of xs is a timestep, each column an example; likewise for ys (must be padded)
        xs = T.imatrix('xs')
        ys = T.imatrix('ys')
            
        cost = self.symbolic_f_prop(xs, ys)
        new_dparams = self.symbolic_b_prop(cost)

        return function([xs, ys], [cost] + new_dparams, allow_input_downcast=True)
            

    def both_prop(self, xs, ys):

        """Like f_prop, but also returns updates for bprop"""
        # return self.both_prop_compiled(xs, ys + [self.out_end])
        return self.both_prop_compiled(xs, ys)
        

    def symbolic_generate(self, xs):
        # xs = np.array(xs)
        # xs = np.reshape(xs, [-1, 1])
        ch = self.encoder.symbolic_f_prop(xs)
        ys = self.decoder.symbolic_generate(ch)
        return ys

    def compile_generate(self):
        xs = T.imatrix('xs')
        return function([xs], self.symbolic_generate(xs), allow_input_downcast=True)

    def generate_answer(self, xs):
        xs = np.array(xs)
        xs = xs.reshape([-1, 1])
        ys_full = (self.generate_function(xs)).reshape([-1]).tolist()
        # I'm lazy and letting theano compute the whole thing, then I take from it
        ys = list(takewhile(lambda x: x != self.out_end, ys_full))
        return ys

    def update_params(self, dec_enc_new_dparams, update_rule):
        """Updates params of both decoder and encoder according to deltas given"""

        num_dec_params = len(self.decoder.params)

        # Decoder
        for index, dparam in enumerate(dec_enc_new_dparams[:num_dec_params]):
            if update_rule == 'sgd':
                self.decoder.params[index].set_value(self.decoder.params[index].get_value() - self.alpha * dparam)
            elif update_rule == 'momentum':
                v_prev = self.decoder.vparams[index]
                v = v_prev*self.mu  - self.alpha * dparam

                self.decoder.params[index].set_value(self.decoder.params[index].get_value() + v)
                self.decoder.vparams[index] = v

        # Encoder
        for index, dparam in enumerate(dec_enc_new_dparams[num_dec_params:]):
            if update_rule == 'sgd':
                self.encoder.params[index].set_value(self.encoder.params[index].get_value() - self.alpha * dparam)
            elif update_rule == 'momentum':

                v_prev = self.encoder.vparams[index]
                v = v_prev*self.mu  - self.alpha * dparam

                self.encoder.params[index].set_value(self.encoder.params[index].get_value() + v)
                self.encoder.vparams[index] = v


    def process_batch(self, all_xs, all_ys, shouldUpdate = True, update_rule='sgd'):
        """Worry about end token and padding to use this function!
        Also this function does all the updates for you"""
        assert(len(all_xs) > 0)
        # or else just return 0 without updating

        # all_dparams = []
        # tot_cost = 0.0
        batch_size = all_xs.shape[1]
        # for xs, ys in zip(all_xs, all_ys):
        cost_and_dparams = self.both_prop(all_xs, all_ys)
        tot_cost = cost_and_dparams[0]
        dparams = [dparam/float(batch_size) for dparam in cost_and_dparams[1:]]
        
        # all_dparams.append(dparams)
        # tot_cost += cost
        
        # n_dparams = len(dparams)
        # dparams_avg = [sum(all_dparams[j][i] for j in xrange(batch_size))/float(batch_size) for i in xrange(n_dparams)]

        # Regularization
        e_reg_updates, e_reg_cost = self.encoder.reg_updates_cost()
        d_reg_updates, d_reg_cost = self.decoder.reg_updates_cost()

        dparams_tot = [(avg + reg) for (avg, reg) in zip(dparams, d_reg_updates + e_reg_updates)]

        if shouldUpdate:
            self.update_params(dparams_tot, update_rule)
            # self.update_params(d_reg_updates + e_reg_updates)
        
        final_cost = (float(tot_cost) / batch_size) + e_reg_cost + d_reg_cost
        
        return final_cost

    def sgd(self, batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True, update_rule='sgd', filename='models/tmp.p'):
        """Implentation of minibatch SGD over all training data (copied from enc_dec). End-tokens will be automatically added later"""
        Y_train = self.pad_data(Y_train)
        if Y_dev is not None:
            Y_dev = self.pad_data(Y_dev)

        # partitions is a list of 2D lists, one for each input length
        partitionX, partitionY = self.partition_XY(X_train, Y_train)

        # Weights for random sampling, given by how many examples are there for each size
        size_probs = np.array([len(size_matrix) for size_matrix in partitionX])
        size_probs = size_probs/float(np.sum(size_probs))
        print "Sum of Probs:", np.sum(size_probs)
        print 'Training:'
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

                # X_batch = X_train[batch_mask] # this notation only works
                # Y_batch = Y_train[batch_mask] # for numpy arrays (not lists)
                # X_batch = list_mask(curr_subset, batch_mask)
                # Y_batch = list_mask(Y_train, batch_mask)
                avg_cost = self.process_batch(X_batch, Y_batch, shouldUpdate = True, update_rule=update_rule) # takes care of sgd

            # Print progress
            if verbose and (epoch % 10) == 0:
                self.save_model(filename)
                print "Epoch", epoch
                tot_cost = 0.0
                for i in range(min(51,N)):
                    single_X = np.array(X_train[i]).reshape([-1, 1])
                    single_Y = np.array(Y_train[i]).reshape([-1, 1])
                    tot_cost += self.process_batch(single_X, single_Y, shouldUpdate = False)
                print "Training Cost (estimate):", tot_cost/51.0
                if X_dev is not None:
                    tot_cost = 0.0
                    for i in range(min(52,N)):
                        single_X = np.array(X_dev[i]).reshape([-1, 1])
                        single_Y = np.array(Y_dev[i]).reshape([-1, 1])
                        tot_cost += self.process_batch(single_X, single_Y, shouldUpdate = False)
                    print "Dev Cost:", tot_cost/52.0

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


    def pad_data(self, Y_train):
        # Pads all lists in Y_train with -1 to have same length
        max_len = max(len(ylist) for ylist in Y_train)
        Y_train_padded = [self.pad_list(max_len, single_list) for single_list in Y_train]
        return Y_train_padded
       
    def pad_list(self, max_len, single_list):
        # Pads a single list with -1 to reach a max len; also adds end token
        padding_needed = max_len - len(single_list)
        new_list = single_list + [self.out_end]
        new_list += [-1 for _ in range(padding_needed)]
        return new_list

    def save_model(self, file_name):
        # Save encoder/decoder to a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'wb') as f:
            pickle.dump((self.encoder, self.decoder), f)

    def load_model(self, file_name):
        # Load encoder/decoder from a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'rb') as f:
            enc, dec = pickle.load(f)
            for self_param, saved_param in zip(self.encoder.params, enc.params):
                self_param.set_value(saved_param.get_value())
            for self_param, saved_param in zip(self.decoder.params, dec.params):
                self_param.set_value(saved_param.get_value())




if __name__ == '__main__':
    new = NewEncDec(12, 12, 12, 12)
    
    X_train = [[1,2,3], [4,5]]
    Y_train = [[7,7,7], [2,3]]
    new.sgd(5, 1000, X_train, Y_train, update_rule='momentum')

    answer = new.generate_answer([1,2,3])
    print 'answer:', answer
    answer2 = new.generate_answer([4,5])
    print 'answer2:', answer2
    
