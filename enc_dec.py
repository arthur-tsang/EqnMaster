import numpy as np
from enc import Encoder
from dec import Decoder

# This class combines the encoder and decoder; it automatically adds
# start and end tokens to the vocabulary.
class EncDec:

    def __init__(self, vdim, hdim, wdim, outdim,
                 alpha=0.005, rho = 0.0001, rseed=10):

    	# Increase the size of the vocabulary to include start and end tokens
    	# Note that vdim is size of encoder's output, whereas outdim is size
    	# of decoder's output (doesn't include +, *,- etc.)

    	vdim = vdim + 2
    	outdim = outdim + 1
    	self.encoder = Encoder(vdim, hdim, wdim, alpha=alpha, rho=rho, rseed=rseed)
    	self.decoder = Decoder(hdim, outdim, alpha=alpha, rho=rho, rseed=rseed)

    	# Positions in the vocabs for start/end tokens
    	self.v_start = vdim - 2
    	self.v_end = vdim - 1
    	self.out_end = outdim - 1


    def f_prop(self, xs, ys):
    	#Forward propagation through encoder and decoder

    	# Assume that xs does not contain start/end tokens
    	enc_xs = np.concatenate(([self.v_start], xs))
    	enc_ys = np.concatenate((xs, [self.v_end]))
    	enc_hidden, enc_cost = self.encoder.f_prop(enc_xs, enc_ys)
    	# print enc_hidden.shape

    	dec_ys = np.concatenate((ys, [self.out_end]))
    	dec_cost = self.decoder.f_prop(dec_ys, enc_hidden)

    	return dec_cost + enc_cost


    def b_prop(self, xs, ys):
    	#Backward propagation through decoder and encoder
    	dec_ys = np.concatenate((ys, [self.out_end]))
    	delta_decoder = self.decoder.b_prop(dec_ys)

    	enc_xs = np.concatenate(([self.v_start], xs))
    	enc_ys = np.concatenate((xs, [self.v_end]))
    	self.encoder.b_prop(enc_xs, enc_ys, delta_decoder)


    def process_batch(self, all_xs, all_ys):

    	# Make sure all gradients are initialized to zero
        for key in self.encoder.params:
            self.encoder.grads[key] = np.zeros(self.encoder.params[key].shape)

        for key in self.decoder.params:
            self.decoder.grads[key] = np.zeros(self.decoder.params[key].shape)    	

        # Compute total cost
    	tot_cost = 0.0
        batch_size = len(all_xs)
        for xs, ys in zip(all_xs, all_ys):
            cost = self.f_prop(xs, ys)
            tot_cost += cost
            self.b_prop(xs, ys)

        # every time we ran backprop, we just added to the gradients,
        # so here we divide by batch_size
        self.encoder.divide_grads(batch_size)
        self.decoder.divide_grads(batch_size)
        
        enc_reg_cost = self.encoder.regularize()
        dec_reg_cost = self.decoder.regularize()

        avg_cost = tot_cost/batch_size
        avg_cost += (enc_reg_cost + dec_reg_cost)

        return avg_cost


    def sgd(self, batch_size, n_epochs, X_train, Y_train, X_dev=None, Y_dev=None, verbose=True):
        # Implentation of SGD over all training data

        N = len(X_train)
        iterations_per_epoch = N / batch_size # using SGD

        # 1 epoch is 1 pass over training data
        for epoch in xrange(n_epochs):

            # For every sub-iteration
            for i in xrange(iterations_per_epoch):

                # Sample a batch
                batch_mask = np.random.choice(N, batch_size)
                X_batch = X_train[batch_mask]
                Y_batch = Y_train[batch_mask]
                avg_cost = self.process_batch(X_batch, Y_batch)

                # Update with SGD
                self.encoder.update_parameters()
                self.decoder.update_parameters()

            # Print progress
            if verbose:
                print "Epoch", epoch
                print "Training Cost:", self.process_batch(X_train, Y_train)
                print "Dev Cost:", self.process_batch(X_dev, Y_dev)


    def grad_check(self, X, Y):
        # Grad-check of encoder decoder

        # TODO: make it robust to single-example X and Y
        h = 1e-5
        X = np.ndarray.astype(X, np.float)
        self.process_batch(X, Y) # sets encoder.grads and decoder.grads
        params = {}
        for key in self.encoder.params:
        	params['e_' + key] = self.encoder.params[key]

        for key in self.decoder.params:
        	params['d_' + key] = self.decoder.params[key]

        grads = {}
        for key in self.encoder.grads:
        	grads['e_' + key] = self.encoder.grads[key]

        for key in self.decoder.grads:
        	grads['d_' + key] = self.decoder.grads[key]

        for key in params:
            print 'Gradient check for ', key
            it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                old_val = it[0].copy()
                it[0] = old_val - h
                low_cost = self.process_batch(X, Y)
                it[0] = old_val + h
                high_cost = self.process_batch(X, Y)
                it[0] = old_val
                num_grad = float(high_cost - low_cost)/(2*h)
                diff = grads[key][it.multi_index] - num_grad
                if abs(diff) > 1e-4:
                    print "Grad Check Failed -- error:", diff
                    print "Numerical gradient:",num_grad,"predicted grad",grads[key][it.multi_index]
                it.iternext()
        print "Grad Check Finished!"

    # Save encoder/decoder to a file (Note that we assume that we remember
    # start/end tokens are at the end of vocabs)
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.save((self.encoder, self.decoder), f)

    # Load encoder/decoder from a file (Note that we assume that we remember
    # start/end tokens are at the end of vocabs)
    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            self.encoder, self.decoder = pickle.load(f)
