#!/usr/bin/env python

import numpy as np

from enc_dec import EncDec
#from misc import get_data
#from visualize_vecs import svd_visualize, tsne_visualize, pca_visualize

from run_helpers import decode, model_solve, preprocess_data, outvocab, invocab

def ed_solve(ed, in_string):
    # TODO: this function is redundant... change eval's call(s) to it
    return model_solve(ed, in_string)

if __name__ == '__main__':

    #Ordering of arguments: file name, data type, hdim, wdim, batch size, epochs, data size, lr, reg, retrain

    ## Filename to save EncDec
    # model_filename = 'models/encdec_test.p'
    model_filename = 'models/' + sys.argv[1]

    ## Data
    # train_file = 'data/3dig_train.p'
    # dev_file = 'data/3dig_dev.p'
    train_file = 'data/' + sys.argv[2] + '_train.p'
    dev_file = 'data/' + sys.argv[2] + '_dev.p'
    X_train, Y_train = preprocess_data(train_file, asNumpy = False)
    X_dev, Y_dev = preprocess_data(dev_file, asNumpy=False)

    ## Hyperparameters
    # hdim = 5
    # wdim = 5
    hdim = int(sys.argv[3])
    wdim = int(sys.argv[4])
    outdim = len(outvocab)
    vdim = len(invocab)
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

    print "Training Enc-Dec"
    print "hdim: %d wdim: %d lr: %f reg: %f epochs: %d batch_size: %d" % (hdim, wdim, alpha, rho, n_epochs, batch_size)
    print "Num Examples: %d" % (dataset_size)
    print "Data: " + train_file
    print "Saving to " + model_filename
    ed = EncDec(vdim, hdim, wdim, outdim, alpha=alpha, rho = rho)

    if sys.argv[10] == 'retrain':
        print 'Retraining'
        ed.load_model(model_filename) # if retraining
    ed.sgd(batch_size, n_epochs, X_train, Y_train, X_dev=X_dev, Y_dev=Y_dev, verbose=True, filename=model_filename)
    ed.save_model(model_filename)

    # EncDec model test
    toy_problems = [decode(x, invocab) for x in X_train]
    
    # L = led.encoder.params['L']
    # #svd_visualize(np.transpose(L), invocab, outfile = 'figs/svd_lstm.jpg')
    # #pca_visualize(np.transpose(L), invocab, outfile = 'figs/pca_lstm.jpg')

    for toy in toy_problems:
        print toy,'=',model_solve(gru, toy)
