import numpy as np
import pickle

import theano.tensor as T
from theano import function

from lstm_enc import LSTMEnc
from lstm_dec import LSTMDec

# # For debugging:
from theano import config
config.floatX = 'float32'
# config.optimizer = 'fast_compile'
# #config.exception_verbosity = 'high'


class LSTMEncDec:
    def __init__(self, vdim, hdim, wdim, outdim, alpha=.005, rho=.0001, rseed = 10):
        # dimensions
        self.vdim = vdim
        self.hdim = hdim
        self.wdim = wdim
        self.outdim = outdim + 1
        self.out_end = self.outdim - 1 # idx of end token

        # others (I don't think we'll need these, save for saving/loading)
        self.alpha = alpha
        self.rho = rho
        self.rseed = rseed

        # sub-models
        self.encoder = LSTMEnc(self.vdim, self.hdim, self.wdim, alpha=alpha, rho=rho, rseed=rseed)
        self.decoder = LSTMDec(self.hdim, self.outdim, alpha=alpha, rho=rho, rseed=rseed)
        
        # compiled functions
        print 'about to compile'
        self.both_prop_compiled = self.compile_both()
        self.generate_function = self.compile_generate()
        print 'done compiling'

    def symbolic_f_prop(self, xs, ys):
        hidden_inter = self.encoder.symbolic_f_prop(xs, np.zeros(2*self.hdim))
        cost = self.decoder.symbolic_f_prop(ys, hidden_inter)
        return cost

    def symbolic_b_prop(self, cost):
        dec_new_dparams = self.decoder.symbolic_b_prop(cost)
        enc_new_dparams = self.encoder.symbolic_b_prop(cost)

        return dec_new_dparams + enc_new_dparams # python-list concatenation

    def compile_both(self):
        """Compiles a function that computes both cost and deltas at the same time"""
        xs = T.ivector('xs')
        ys = T.ivector('ys')
        
        cost = self.symbolic_f_prop(xs, ys)
        new_dparams = self.symbolic_b_prop(cost)

        return function([xs, ys], [cost] + new_dparams, allow_input_downcast=True)

    def both_prop(self, xs, ys):
        """Like f_prop, but also returns updates for bprop"""
        return self.both_prop_compiled(xs, ys + [self.out_end])
        

    def symbolic_generate(self, xs):
        ch = self.encoder.symbolic_f_prop(xs, np.zeros(2*self.hdim))
        ys = self.decoder.symbolic_generate(ch)
        return ys

    def compile_generate(self):
        xs = T.ivector('xs')
        f = function([xs], self.symbolic_generate(xs))
        return f

    def generate_answer(self, xs):
        return self.generate_function(xs)


    def update_params(self, dec_enc_new_dparams):
        """Updates params of both decoder and encoder according to deltas given"""
        for param, dparam in zip(self.decoder.params + self.encoder.params, dec_enc_new_dparams):
            param.set_value(param.get_value() - self.alpha * dparam)

    def process_batch(self, all_xs, all_ys, shouldUpdate = True):
        """Don't worry about end token to use this function!
        Also this function does all the updates for you"""
        assert(len(all_xs) > 0)
        # or else just return 0 without updating

        all_dparams = []
        tot_cost = 0.0
        batch_size = len(all_xs)
        for xs, ys in zip(all_xs, all_ys):
            cost_and_dparams = self.both_prop(xs, ys)
            cost = cost_and_dparams[0]
            dparams = cost_and_dparams[1:]
            
            all_dparams.append(dparams)
            tot_cost += cost
        
        n_dparams = len(all_dparams[0])
        dparams_avg = [sum(all_dparams[j][i] for j in xrange(batch_size))/float(batch_size) for i in xrange(n_dparams)]

        # Regularization
        e_reg_updates, e_reg_cost = self.encoder.reg_updates_cost()
        d_reg_updates, d_reg_cost = self.decoder.reg_updates_cost()

        if shouldUpdate:
            self.update_params(dparams_avg)
            self.update_params(d_reg_updates + e_reg_updates)
        
        final_cost = float(tot_cost) / batch_size + e_reg_cost + d_reg_cost
        
        return final_cost

    def sgd(self, batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True):
        """Implentation of minibatch SGD over all training data (copied from enc_dec)"""

        print 'Training:'
        print 'Train Set Size:', len(Y_train)
        # Helpers
        def list_mask(full_list, mask):
            # extracts indices from the full_list as per the mask
            return [full_list[idx] for idx in mask]

        # Actual code
        N = len(X_train)
        iterations_per_epoch = 1+ N / batch_size # using SGD

        # 1 epoch is 1 pass over training data
        for epoch in xrange(n_epochs):
            # For every sub-iteration
            for i in xrange(iterations_per_epoch):
                # print i
                # Sample a batch
                batch_mask = np.random.choice(N, batch_size)
                # X_batch = X_train[batch_mask] # this notation only works
                # Y_batch = Y_train[batch_mask] # for numpy arrays (not lists)
                X_batch = list_mask(X_train, batch_mask)
                Y_batch = list_mask(Y_train, batch_mask)
                avg_cost = self.process_batch(X_batch, Y_batch, shouldUpdate = True) # takes care of sgd

            # Print progress
            if verbose and (epoch % 10) == 0:
                print "Epoch", epoch
                print "Training Cost (estimate):", self.process_batch(X_train[:50], Y_train[:50], shouldUpdate = False)
                if X_dev is not None:
                    print "Dev Cost:", self.process_batch(X_dev, Y_dev, shouldUpdate = False)

    def save_model(self, file_name):
        # Save encoder/decoder to a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'wb') as f:
            pickle.dump((self.encoder, self.decoder), f)

    def load_model(self, file_name):
        # Load encoder/decoder from a file (Note that we assume that we remember
        # start/end tokens are at the end of vocabs)
        with open(file_name, 'rb') as f:
            self.encoder, self.decoder = pickle.load(f)



if __name__ == '__main__':
    lstm = LSTMEncDec(12,12,12,12)

    print 'processing batch'
    cost1 = lstm.process_batch([[3,2]],[[0,0]])
    cost2 = lstm.process_batch([[1,2,3]],[[2,2,2]])
    cost3 = lstm.process_batch([[1,2,3],[2,2,2]], [[3,2],[0,0]])
    print cost1, cost2, cost3
    print lstm.generate_answer([1,2,3])
    lstm.sgd(2, 2, [[1,2,3], [4,5], [3,4]], [[3,3], [2], [1]])
    print 'all done'
