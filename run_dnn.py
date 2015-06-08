#!/usr/bin/env python

import numpy as np
import sys
import signal
from d_nn import DNN
#from visualize_vecs import svd_visualize, tsne_visualize, pca_visualize

from run_helpers2 import extract_discr_data, invocab2, decode

if __name__ == '__main__':



    #Ordering of arguments: file name, data type, hdim, wdim, batch size, epochs, data size, lr, reg, retrain

    ## Filename to save LSTMEncDec
    # model_filename = 'models/gru_test.p'
    model_filename = 'models/' + sys.argv[1]

    ## Data
    # train_file = 'data/3dig_train.p'
    # dev_file = 'data/3dig_dev.p'
    train_file = 'data/' + sys.argv[2] + '_train.p'
    dev_file = 'data/' + sys.argv[2] + '_dev.p'
    X_train, Y_train = extract_discr_data(train_file, asNumpy = False)
    X_dev, Y_dev = extract_discr_data(dev_file, asNumpy=False)

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        toy_problems = [decode(x, invocab2) for x in X_train]
        for toy, x in zip(toy_problems, X_train):
            print toy,'=', drnn.generate_answer(x)
        sys.exit(0)

    
    signal.signal(signal.SIGINT, signal_handler)

    ## Hyperparameters
    # hdim = 5
    # wdim = 5
    hdim = int(sys.argv[3])
    wdim = int(sys.argv[4])
    # outdim = len(outvocab)
    outdim = 2
    vdim = len(invocab2)
    # batch_size = 100
    # n_epochs = 1000
    batch_size = int(sys.argv[5])
    n_epochs = int(sys.argv[6])

    # dataset_size = 1000
    dataset_size = int(sys.argv[7])
    X_train = X_train[:dataset_size]
    Y_train = Y_train[:dataset_size]
    
    ## EncDec model train
    # alpha = 0.01
    # rho = 0.0000
    alpha = float(sys.argv[8])
    rho = float(sys.argv[9])

    print "Training D-RNN"
    print "hdim: %d wdim: %d lr: %f reg: %f epochs: %d batch_size: %d" % (hdim, wdim, alpha, rho, n_epochs, batch_size)
    print "Num Examples: %d" % (dataset_size)
    print "Data: " + train_file
    print "Saving to " + model_filename
    dnn = DNN(vdim, hdim, wdim, outdim, alpha=alpha, rho = rho)

    if sys.argv[10] == 'retrain':
        print 'Retraining'
        dnn.load_model(model_filename) # if retraining
    dnn.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=X_dev, Y_dev=Y_dev, verbose=True, update_rule='momentum', filename=model_filename)
    #dnn.save_model(model_filename)

    # ## LSTMEncDec model test
    toy_problems = [decode(x, invocab2) for x in X_train[:50]]
    
    # L = led.encoder.params['L']
    # #svd_visualize(np.transpose(L), invocab, outfile = 'figs/svd_lstm.jpg')
    # #pca_visualize(np.transpose(L), invocab, outfile = 'figs/pca_lstm.jpg')

    for toy, x in zip(toy_problems, X_train):
        print toy,'=', dnn.generate_answer(x)
