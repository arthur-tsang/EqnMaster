import numpy as np

import theano.tensor as T
from theano import function

from lstm_enc import LSTMEnc
from lstm_dec import LSTMDec


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

        return function([xs, ys], [cost] + new_dparams)

    def both_prop(self, xs, ys):
        """Like f_prop, but also returns updates for bprop"""
        return self.both_prop_compiled(xs, ys + [self.out_end])
        
    # TODO: write a generator

    def update_params(self, dec_enc_new_dparams):
        """Updates params of both decoder and encoder according to deltas given"""
        for param, dparam in zip(self.decoder.params + self.encoder.params, dec_enc_new_dparams):
            param.set_value(param.get_value() + dparam)

    def process_batch(self, all_xs, all_ys):
        assert(len(all_xs) > 0)
        # or else just return 0

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

        self.update_params(dparams_avg)
        self.update_params(d_reg_updates + e_reg_updates)
        
        final_cost = float(tot_cost) / batch_size + e_reg_cost + d_reg_cost
        
        return final_cost

if __name__ == '__main__':
    lstm = LSTMEncDec(12,12,12,12)

    print 'processing batch'
    cost = lstm.process_batch([[3,2]],[[0,0]])
    cost = lstm.process_batch([[1,2,3]],[[2,2,2]])
    cost = lstm.process_batch([[1,2,3],[2,2,2]], [[3,2],[0,0]])
    print cost
    print 'all done'
