#!/usr/bin/env python

import numpy as np

from gru_encdec import GRUEncDec
#from visualize_vecs import svd_visualize, tsne_visualize, pca_visualize

from run_helpers import decode, model_solve, preprocess_data, outvocab, invocab

if __name__ == '__main__':

    ## Filename to save LSTMEncDec
    model_filename = 'models/gru_1000.p'

    ## Data
    X_train, Y_train = preprocess_data('data/3dig_train.p', asNumpy = False)
    X_dev, Y_dev = preprocess_data('data/3dig_dev.p', asNumpy=False)

    ## Hyperparameters
    hdim = 50
    wdim = 50
    outdim = len(outvocab)
    vdim = len(invocab)
    batch_size = 5
    n_epochs = 5000

    # X_train = X_train[:100]
    # Y_train = Y_train[:100]

    # X_train = X_train[:20]
    # Y_train = Y_train[:20]

    # X_train = [[4,5,1,10,11,10,8,2,6]]
    # Y_train = [[1,2,7,7]]


    X_train = X_train[:1000]
    Y_train = Y_train[:1000]
    
    ## EncDec model train
    gru = GRUEncDec(vdim, hdim, wdim, outdim, alpha=0.01, rho = 0.0000)

    #gru.load_model(model_filename) # if retraining
    gru.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True)
    gru.save_model(model_filename)

    # ## LSTMEncDec model test
    # toy_problems = [decode(x, invocab) for x in X_train]
    # # toy_problems = ['5+15','17+98','7+7','3+7']
    
    # L = led.encoder.params['L']
    # #svd_visualize(np.transpose(L), invocab, outfile = 'figs/svd_lstm.jpg')
    # #pca_visualize(np.transpose(L), invocab, outfile = 'figs/pca_lstm.jpg')

    # for toy in toy_problems:
    #     print toy,'=',model_solve(led, toy)
