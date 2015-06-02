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
        self.outdim = outdim

        # others (I don't think we'll need these, save for saving/loading)
        self.alpha = alpha
        self.rho = rho
        self.rseed = rseed

        # sub-models
        self.encoder = LSTMEnc(vdim, hdim, wdim, alpha=alpha, rho=rho, rseed=rseed)
        self.decoder = LSTMDec(hdim, outdim, alpha=alpha, rho=rho, rseed=rseed)
        
        # compiled functions
        #self.f_prop_function = compile_f_prop()
        print 'about to compile'
        self.both_prop_compiled = self.compile_both()
        print 'done compiling'

    def symbolic_f_prop(self, xs, ys):
        hidden_inter = self.encoder.symbolic_f_prop(xs, np.zeros(2*self.hdim)) #note: are we allowed to put np.zeros here? (ipython makes me think so)
        cost = self.decoder.symbolic_f_prop(ys, hidden_inter)
        return cost

    # def compile_f_prop(self):
    #     xs = T.ivector('xs')
    #     ys = T.ivector('ys')
        
    #     return function([xs, ys], self.symbolic_f_prop(xs, ys))
        
    # def f_prop(self, xs, ys):
    #     # TODO: take care of end token
    #     return self.f_prop_function(xs, ys)

    def symbolic_b_prop(self, cost):
        dec_new_dparams = self.decoder.symbolic_b_prop(cost)
        enc_new_dparams = self.encoder.symbolic_b_prop(cost)

        # print 'symbolic bprop shapes', dec_new_dparams[0].shape

        return dec_new_dparams + enc_new_dparams # python-list concatenation

    # def regularization_cost(self):
    #     return 

    def compile_both(self):
        """Compiles a function that computes both cost and deltas at the same time"""
        xs = T.ivector('xs')
        ys = T.ivector('ys')
        
        cost = self.symbolic_f_prop(xs, ys)
        new_dparams = self.symbolic_b_prop(cost)

        return function([xs, ys], [cost] + new_dparams)
        

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
            cost_and_dparams = self.both_prop_compiled(xs, ys)
            cost = cost_and_dparams[0]
            dparams = cost_and_dparams[1:]
            
            all_dparams.append(dparams)
            tot_cost += cost
        
        n_dparams = len(all_dparams[0])
        dparams_avg = [sum(all_dparams[j][i] for j in xrange(batch_size))/float(batch_size) for i in xrange(n_dparams)]
        

        # print 'shape avant', type(all_dparams[0][0]), all_dparams[0].shape
        # dparams = [np.average(dp, axis=0) for dparams in all_dparams \
        #            for dparam in dparams]
        # print 'shape apres', dparams.shape
        self.update_params(dparams)

        # forget about regularizing for now

        return 1. * tot_cost / batch_size


    # actual def f_prop will worry about end-tokens

# TODO: code up
if __name__ == '__main__':
    lstm = LSTMEncDec(12,12,12,12)

    print 'processing batch'
    cost = lstm.process_batch([[1,2,3]],[[2,2,2]])
    # cost = lstm.process_batch([[1,2,3],[2,2,2]], [[3,2],[0,0]])
    print cost
    print 'all done'
