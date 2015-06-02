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
        self.f_prop_function = compile_f_prop()

    def symbolic_f_prop(self, xs, ys):
        hidden_inter = self.encoder.symbolic_f_prop(xs, np.zeros(2*self.hdim)) #note: are we allowed to put np.zeros here? (ipython makes me think so)
        cost = self.decoder.symbolic_f_prop(ys, hidden_inter)
        return cost

    def compile_f_prop(self):
        xs = T.ivector('xs')
        ys = T.ivector('ys')
        
        return function([xs, ys], self.symbolic_f_prop(xs, ys))

    def symbolic_b_prop(self, cost):
        dec_new_dparams = self.decoder.symbolic_b_prop(cost)
        enc_new_dparams = self.encoder.symbolic_b_prop(cost)
        return (dec_new_dparams, enc_new_dparams)


    # actual def f_prop will worry about end-tokens

# TODO: code up
if __name__ == '__main__':
    lstm = LSTMEncDec(12,12,12,12)
    print 'all done'
