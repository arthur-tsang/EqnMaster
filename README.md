# cs224d
Neural Arithmetic and Equation Solver

## Summary
In this project, we explore the use of RNNs, LSTMs, and the like to solve arithmetic problems. So far, we have a baseline and simple rnnlm model set up for 3-digit addition, as well as a discriminative test if an equation involving 3-digit addition is correct. In addition, we've implemented an RNN encoder-decoder model.

## How to run
Right now the code is in a bit of a mess, but you can run `naive_rnnlm.py`. Use `naive_rnnlm.py train` to train, and no argument for a demo on a couple problems. Currently the only way to change hyperparameters is to change the actual python file, but we'd like to change that eventually.

To "train" and demo the baseline, just run `baseline.py`.

To train and demo the Encoder-Decoder, run `run_enc_dec.py`. This file is definitely subject to change.

The real evaluation takes place in `eval.py` which will be back in operation shortly.

## To-do list
* Learn Theano
* Write encoder-decoder LSTM
* Clean up `eval.py`
* Try training digit-specific LSTMs (more of a tangent)

## Dependencies
The only external dependency we're using right now is `numpy`, though we plan to integrate `theano` in the near future.
