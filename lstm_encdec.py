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
        return [dec_new_dparams, enc_new_dparams]

    def compile_both(self):
        xs = T.ivector('xs')
        ys = T.ivector('ys')
        
        cost = self.symbolic_f_prop(xs, ys)
        print 'time 0'
        new_dparams = self.symbolic_b_prop(cost)
        print 'time 1'

        return function([xs,ys], new_dparams) # TODO: remove this line
        'forward prop compiles'


        return function([xs, ys], [cost, new_dparams])
        

    def update_params(dec_enc_new_dparams):
        dec_new_dparams, enc_new_dparams = dec_enc_new_dparams
        for param, dparam in zip(self.decoder.params, dec_new_dparams) + zip(self.encoder.params, enc_new_params):
            param.set_value(param.get_value() + dparam)

    def process_batch(self, all_xs, all_ys):
        
        # def sum_dparams(dparams1, dparams2):
        #     dec_dparams1, enc_dparams1 = dparams1
        #     dec_dparams2, enc_dparams2 = dparams2
        #     dec_dparams = [a + b for a,b in zip(dec_dparams1, dec_dparams2)]
        #     enc_dparams = [a + b for a,b in zip(enc_dparams1, enc_dparams2)]
        #     return (dec_dparams, enc_dparams)

        all_dec_dparams = []
        all_enc_dparams = []
        tot_cost = 0.0
        batch_size = len(all_xs)
        for xs, ys in zip(all_xs, all_ys):
            cost, (dec_dparams, enc_dparams) = self.both_prop_compiled(xs, ys)
            
            all_dec_dparams.append(dec_dparams)
            all_enc_dparams.append(enc_dparams)
            tot_cost += cost
        
        dec_dparams = np.average(all_dec_dparams, axis=0)
        enc_dparams = np.average(all_enc_dparams, axis=0)
        update_params((dec_dparams, enc_dparams))

        # forget about regularizing for now

        return 1. * tot_cost / batch_size


    # actual def f_prop will worry about end-tokens

# TODO: code up
if __name__ == '__main__':
    lstm = LSTMEncDec(12,12,12,12)

    # print 'processing batch'
    # cost = lstm.process_batch([[1,2,3],[2,2,2]], [[3,2],[0,0]])
    # print cost
    # print 'all done'
