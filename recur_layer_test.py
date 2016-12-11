#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import lasagne
from scipy.linalg import sqrtm # sqrt for matrix

_M = 3
N_BATCH = 1

N_TIME_STEPS = 50
N_RECURRENT = 3
LEARNING_RATE = .01
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 15


def gen_data():
    
    
    return

def main(num_epochs=NUM_EPOCHS):
    print "Building network ..."

    w_in_to_hid = np.eye(3)
    w_hid_to_hid = np.eye(3) / 2
    w_out = np.eye(3)

    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, _M))
    l_rnn = lasagne.layers.RecurrentLayer(
        incoming=l_in,
        num_units=N_RECURRENT,
        W_in_to_hid=w_in_to_hid,
        W_hid_to_hid=w_hid_to_hid,
        b=None,
        nonlinearity=lasagne.nonlinearities.rectify
    )
    l_slice = lasagne.layers.SliceLayer(
        incoming=l_rnn,
        indices=-1,
        axis=1
    )
    l_out = lasagne.layers.DenseLayer(
        incoming=l_slice,
        num_units=3,
        W=w_out,
        b=None,
        nonlinearity=lasagne.nonlinearities.identity
    )

    AA = np.random.uniform(low=-1, high=1, size=(3, 3))
    X = np.zeros(shape=(N_BATCH, N_TIME_STEPS, _M))        
    tmp = np.random.uniform(low=0, high=1, size=(1, _M))
    if np.linalg.norm(tmp) < 1e-6:
        tmp[0] = 1
    tmp = np.dot(tmp, AA)
    tmp /= np.linalg.norm(tmp)
    for j in range(N_TIME_STEPS):
        X[0, j, :] = tmp
    X_sliced = X[:, 0, :]

    network_output = lasagne.layers.get_output(l_out)
    MMM = T.eye(n=3,m=3) - l_rnn.W_hid_to_hid

    M_min_half = T.matrix('M_min_half')

    cost = T.sum(
        (
            T.dot(
                M_min_half,
                np.transpose(X_sliced).astype(theano.config.floatX) - 
                T.dot(MMM, T.transpose(network_output))
            )
        )**2
    )

    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    all_params.pop(0) 
    all_params.pop(1)

    print "Computing updates ..."
    updates = lasagne.updates.sgd(cost, all_params, LEARNING_RATE)

    print "Compiling functions ..."
    train = theano.function(
        [l_in.input_var, M_min_half],
        cost, updates=updates
    )

    compute_cost = theano.function(
        [l_in.input_var, M_min_half], cost
    )

    result = theano.function([l_in.input_var],
                lasagne.layers.get_output(l_out))

    print("cost_")
    MM = np.eye(3) - l_rnn.W_hid_to_hid.get_value()    #shape: (3, 3)
    M_half = sqrtm(MM)   #shape: (3, 3)
    M_minus_half = np.linalg.inv(M_half) #shape: (3, 3)
    print compute_cost(
        X.astype(theano.config.floatX),
        M_minus_half.astype(theano.config.floatX)
    )
    f = open('params.txt', 'a')
    print "Training ..."
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                Z = result(X) #shape: (1, 3)

                W = l_rnn.W_hid_to_hid.get_value()

                # Z = Z.flatten()
                W_old = np.copy(W)

                MM = np.eye(3) - l_rnn.W_hid_to_hid.get_value()    #shape: (3, 3)
                M_half = sqrtm(MM)   #shape: (3, 3)
                M_minus_half = np.linalg.inv(M_half) #shape: (3, 3)

                train(
                    X.astype(theano.config.floatX),
                    M_minus_half.astype(theano.config.floatX) 
                )

                W_new = np.copy(l_rnn.W_hid_to_hid.get_value())
                delta_W = W_new - W_old
                delta_W = 1. / 2 * (delta_W + np.transpose(delta_W))
                l_rnn.W_hid_to_hid.set_value(W_old + delta_W)
                
                
            # y_val = result(X)
            # W = l_rnn.W_hid_to_hid.get_value()
            # MM = np.eye(3) - W    #shape: (3, 3)
            # M_half = sqrtm(MM)   #shape: (3, 3)
            # M_minus_half = np.linalg.inv(M_half)    #shape: (3, 3)
            print "Cost:"
            MM = np.eye(3) - l_rnn.W_hid_to_hid.get_value()    #shape: (3, 3)
            M_half = sqrtm(MM)   #shape: (3, 3)
            M_minus_half = np.linalg.inv(M_half) #shape: (3, 3)
            print compute_cost(
                X.astype(theano.config.floatX),
                M_minus_half.astype(theano.config.floatX)
            )
    except KeyboardInterrupt:
        pass

    f.close()


if __name__ == '__main__':
    main()
