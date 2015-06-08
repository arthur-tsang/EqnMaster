#!/usr/bin/env python

# Here, we evaluate the models according to the dev sets

# This file is temporarily closed for renovation (metaphorically speaking)

import os.path
import pickle
#from run_enc_dec import ed_solve
#from enc_dec import EncDec
#from naive_rnnlm import NaiveRnnlm
#from naive_rnnlm_discr import NaiveRnnlmDiscr
from baseline import BigramBaseline
from misc import lengthen, get_data

from visualize_vecs import svd_visualize, pca_visualize

from gru_encdec import GRUEncDec
from lstm_encdec import LSTMEncDec

from d_rnn import DRNN
from d_gru import DGRU

import run_helpers # doesn't allow '-' signs
import run_helpers2 # allows '-' signs



# def bool_metric(correct, given):
#     return int(correct == given)

# def dig_metric(correct, given):
#     # Evaluation metric: 1/4 point per matching digit
#     # (not strict booleans here, since we're not good enough (yet))
#     y_len = 4
#     correct = lengthen(correct, y_len) # add initial zeros
#     given = lengthen(given, y_len)
#     score = sum(c == g for c,g in zip(correct, given)) / float(y_len)
#     return score

def align_metric(correct, given):
    """Count digit similarities between correct answer and given answer"""

    # right-align our answers to compare
    r_correct = reversed(correct)
    r_given = reversed(given)

    score = sum(int(c == g) for c,g in zip(r_correct, r_given))
    score /= float(len(correct)) # normalize points

    # print 'correct',correct,'given',given

    return score
    
def strict_metric(correct, given):
    return int(correct == given)

def num_metric(correct, given):
    """Count raw distance away from the correct answer"""
    n_correct = int(correct)
    n_given = int(given)
    diff = abs(n_correct - n_given)

    return diff


def eval_model(predict_fn, xy_data, metric = align_metric):
    scores = [metric(y, predict_fn(x)) for x,y in xy_data]
    return 1.0 * sum(scores) / len(scores)


# def nr_test(rnns_file, data, discr = False):
#     print 'Warning: this function might not act right'
#     metric = dig_metric if not discr else bool_metric
#     nr = NaiveRnnlm() if not discr else NaiveRnnlmDiscr()
#     if os.path.exists(rnns_file):
#         with open(rnns_file, 'r') as f:
#             nr.rnns = pickle.load(f)
#         print 'nr at', rnns_file, eval_model(nr.predict_one, data, metric)
#     else:
#         print 'nr at', rnns_file, 'not found'

def discriminative_test():
    discr_add_train = get_data('data/d_add_train.p')
    discr_add_dev = get_data('data/d_add_dev.p')
    discr_subtr_train = get_data('data/d_subtr_train.p')
    discr_subtr_dev = get_data('data/d_subtr_dev.p')
    discr_mult_train = get_data('data/d_mult_train.p')
    discr_mult_dev = get_data('data/d_mult_dev.p')

    # TESTING discriminative stuff

    drnn = DRNN(len(run_helpers2.invocab), 50, 50, len(run_helpers2.outvocab))
    drnn.load_model('models/drnn_mult_full.p')
    drnn_fn = lambda x: run_helpers2.model_solve_discr(drnn, x)
    print 'drnn mult train', eval_model(drnn_fn, discr_mult_train, metric=strict_metric)
    print 'drnn mult dev', eval_model(drnn_fn, discr_mult_dev, metric=strict_metric)

    drnn.load_model('models/drnn_subtr_full.p')
    print 'drnn subtr train', eval_model(drnn_fn, discr_subtr_train, metric=strict_metric)
    print 'drnn subtr dev', eval_model(drnn_fn, discr_subtr_dev, metric=strict_metric)

    dgru = DGRU(len(run_helpers2.invocab), 50, 50, len(run_helpers2.outvocab))
    dgru.load_model('models/dgru_mult_full.p')
    dgru_fn = lambda x: run_helpers2.model_solve_discr(dgru, x)
    print 'dgru mult train', eval_model(dgru_fn, discr_mult_train, metric=strict_metric)
    print 'dgru mult dev', eval_model(dgru_fn, discr_mult_dev, metric=strict_metric)

    dgru.load_model('models/dgru_add_full.p')
    print 'dgru add train', eval_model(dgru_fn, discr_add_train, metric=strict_metric)
    print 'dgru add dev', eval_model(dgru_fn, discr_add_dev, metric=strict_metric)

    dgru.load_model('models/dgru_subtr_full.p')
    print 'dgru subtr train', eval_model(dgru_fn, discr_subtr_train, metric=strict_metric)
    print 'dgru subtr dev', eval_model(dgru_fn, discr_subtr_dev, metric=strict_metric)



if __name__ == '__main__':
    add_train = get_data('data/3dig_train.p')
    add_dev = get_data('data/3dig_dev.p')
    subtr_train = get_data('data/subtr_train.p')
    subtr_dev = get_data('data/subtr_dev.p')
    mult_train = get_data('data/mult_train.p')
    mult_dev = get_data('data/mult_dev.p')

    add4_train = get_data('data/4dig_train.p')
    add4_dev = get_data('data/4dig_dev.p')
    add5_train = get_data('data/5dig_train.p')
    add5_dev = get_data('data/5dig_dev.p')
    add6_train = get_data('data/6dig_train.p')
    add6_dev = get_data('data/6dig_dev.p')
    add7_train = get_data('data/7dig_train.p')
    add7_dev = get_data('data/7dig_dev.p')


    # # TESTING LSTMs
    
    # lstm = LSTMEncDec(len(run_helpers.invocab), 50, 50, len(run_helpers.outvocab))
    # lstm.load_model('models/lstm_add_full.p')
    # lstm_fn = lambda x: run_helpers.model_solve(lstm, x)
    # print 'lstm add train', eval_model(lstm_fn, add_train)
    # print 'lstm add dev', eval_model(lstm_fn, add_dev)
    
    # svd_visualize(lstm.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/lstm_add_svd.jpg', title='Word vectors - LSTM addition')
    

    # lstm.load_model('models/lstm_mult_full.p') # not ready yet
    # # print 'lstm mult train', eval_model(lstm_fn, mult_train)
    # # print 'lstm mult dev', eval_model(lstm_fn, mult_dev)

    # svd_visualize(lstm.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/lstm_mult_svd.jpg', title='Word vectors - LSTM multiplication')


    # lstm = LSTMEncDec(len(run_helpers2.invocab), 50, 50, len(run_helpers2.outvocab))
    # lstm.load_model('models/lstm_subtr_full.p')
    # lstm_fn = lambda x: run_helpers2.model_solve(lstm, x)
    # print 'lstm subtr train', eval_model(lstm_fn, subtr_train)
    # print 'lstm subtr dev', eval_model(lstm_fn, subtr_dev)
    
    # svd_visualize(lstm.encoder.params[0].get_value().T, run_helpers2.invocab, outfile='figs/lstm_subtr_svd.jpg', title='Word vectors - LSTM subtraction')

    # TESTING GRUs
    
    #gru_add = GRUEncDec(len(run_helpers.invocab), 50, 50, len(run_helpers.outvocab))
    #gru_add.load_model('models/gru_add_full.p')
    # gru_add_fn = lambda x: run_helpers.model_solve(gru_add, x)
    # print 'gru add train', eval_model(gru_add_fn, add_train)
    # print 'gru add dev', eval_model(gru_add_fn, add_dev)
    # print 'gru add train strict', eval_model(gru_add_fn, add_train, metric=strict_metric)
    # print 'gru add dev strict', eval_model(gru_add_fn, add_dev, metric=strict_metric)

    # svd_visualize(gru_add.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/gru_add_svd.jpg', title='Word vectors - GRU addition')
    # pca_visualize(gru_add.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/gru_add_pca3d.jpg', title='Word vectors - GRU addition')

    # print 'gru add4 dev', eval_model(gru_add_fn, add4_dev)
    # print 'gru add5 dev', eval_model(gru_add_fn, add5_dev)
    # print 'gru add6 dev', eval_model(gru_add_fn, add6_dev)
    # print 'gru add7 dev', eval_model(gru_add_fn, add7_dev)


    gru_subtr = GRUEncDec(len(run_helpers2.invocab), 50, 50, len(run_helpers2.outvocab))
    gru_subtr.load_model('models/gru_subtr_full.p')
    # gru_subtr_fn = lambda x: run_helpers2.model_solve(gru_subtr, x)
    # print 'gru subtr train', eval_model(gru_subtr_fn, subtr_train)
    # print 'gru subtr dev', eval_model(gru_subtr_fn, subtr_dev)
    # print 'gru subtr train strict', eval_model(gru_subtr_fn, subtr_train, metric=strict_metric)
    # print 'gru subtr dev strict', eval_model(gru_subtr_fn, subtr_dev, metric=strict_metric)

    #svd_visualize(gru_subtr.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/gru_subtr_svd.jpg', title='Word vectors - GRU subtraction')
    
    pca_visualize(gru_subtr.encoder.params[0].get_value().T, run_helpers.invocab, outfile='figs/gru_subtr_pca3d.jpg', title='Word vectors - GRU subtraction')
    
    # discriminative_test()












    # Just ignore what's below this line really

    # train_data = get_data('data/3dig_train.p')
    # dev_data = get_data('data/3dig_dev.p')
    # train_data_discr = get_data('data/neg_train.txt')
    # dev_data_discr = get_data('data/neg_dev.txt')
    # train_data_short = get_data('data/2dig_train.p')
    # dev_data_short = get_data('data/2dig_dev.p')
    

    
    # print 'Dev set scores'
    # ed = EncDec(12,10,10,10) # TODO: remove magic numbers
    # ed.load_model('models/ed_simple.p')
    # ed_fn = lambda x : ed_solve(ed, x)
    # print 'ed dev', eval_model(ed_fn, dev_data_short)
    # ed2 = EncDec(12,10,10,10)
    # ed2.load_model('models/ed_full.p')
    # ed2_fn = lambda x : ed_solve(ed2, x)
    # print 'ed2 dev', eval_model(ed2_fn, dev_data)
    
    

    # # bigram baseline part
    # bb = BigramBaseline()
    # bb.learn(train_data)
    # #print 'bb dev', eval_model(bb.predict_one, dev_data)
    
    # # # naive rnn part
    # # nr_test('rnn_naive.txt', dev_data)
    # # nr_test('rnn_naive_oracle.txt', dev_data)
    # # nr_test('rnn_naive_oracle_bptt.txt', dev_data)
    # # nr_test('rnn_naive_rot.txt', dev_data)
    # # nr_test('rnn_naive_rot_bptt.txt', dev_data)
    # # nr_test('rnn_naive_discr.txt', dev_data_discr, True)


    # print 'Train set scores'
    # print 'ed train', eval_model(ed_fn, train_data_short)
    # print 'ed2 train', eval_model(ed2_fn, train_data) # 3-digit ed

    
    # # bigram baseline part
    #print 'bb train', eval_model(bb.predict_one, train_data)
    
    # # naive rnn part
    # nr_test('rnn_naive.txt', train_data)
    # nr_test('rnn_naive_oracle.txt', train_data)
    # nr_test('rnn_naive_oracle_bptt.txt', train_data)
    # nr_test('rnn_naive_rot.txt', train_data)
    # nr_test('rnn_naive_rot_bptt.txt', train_data)
    # nr_test('rnn_naive_discr.txt', train_data_discr, True)
