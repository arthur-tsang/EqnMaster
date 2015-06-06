#!/usr/bin/env python

import numpy as np
import sys
from lstm_encdec import LSTMEncDec
#from visualize_vecs import svd_visualize, tsne_visualize, pca_visualize

from run_helpers import decode, model_solve, preprocess_data, outvocab, invocab

if __name__ == '__main__':

    #Ordering of arguments: file name, data type, hdim, wdim, batch size, epochs, data size, lr, reg, retrain

    ## Filename to save LSTMEncDec
    # model_filename = 'models/lstm_mult_full.p'
    model_filename = 'models/' + sys.argv[1]

    ## Data

    train_file = 'data/' + sys.argv[2] + '_train.p'
    dev_file = 'data/' + sys.argv[2] + '_dev.p'
    X_train, Y_train = preprocess_data(train_file, asNumpy = False)
    X_dev, Y_dev = preprocess_data(dev_file, asNumpy=False)

    ## Hyperparameters
    # hdim = 15
    # wdim = 15
    hdim = int(sys.argv[3])
    wdim = int(sys.argv[4])
    outdim = len(outvocab)
    vdim = len(invocab)
    # batch_size = 50
    # n_epochs = 1000
    batch_size = int(sys.argv[5])
    n_epochs = int(sys.argv[6])
    # dataset_size = 10000
    dataset_size = int(sys.argv[7])
    X_train = X_train[:dataset_size]
    Y_train = Y_train[:dataset_size]
    
    ## EncDec model train
    # alpha = 0.01
    # rho = 0.0001
    alpha = float(sys.argv[8])
    rho = float(sys.argv[9])

    print "Training LSTM"
    print "hdim: %d wdim: %d lr: %f reg: %f epochs: %d batch_size: %d" % (hdim, wdim, alpha, rho, n_epochs, batch_size)
    print "Num Examples: %d" % (dataset_size)
    print "Data: " + train_file
    print "Saving to " + model_filename
    led = LSTMEncDec(vdim, hdim, wdim, outdim, alpha=alpha, rho = rho)

    if sys.argv[10] == 'retrain':
        print 'Retraining'
        led.load_model(model_filename) # if retraining
    led.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=X_dev, Y_dev=Y_dev, verbose=True, filename=model_filename)
    led.save_model(model_filename)

    # ## LSTMEncDec model test
    toy_problems = [decode(x, invocab) for x in X_train]
    
    # L = led.encoder.params['L']
    # #svd_visualize(np.transpose(L), invocab, outfile = 'figs/svd_lstm.jpg')
    # #pca_visualize(np.transpose(L), invocab, outfile = 'figs/pca_lstm.jpg')

    for toy in toy_problems:
        print toy,'=',model_solve(led, toy)
