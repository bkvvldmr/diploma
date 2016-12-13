#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import lasagne
from scipy.linalg import sqrtm # sqrt for matrix
from numpy import linalg as LA

_M = 3
N_BATCH = 1

N_TIME_STEPS = 50
N_RECURRENT = 3
LEARNING_RATE = .01
# How often should we check the output?
EPOCH_SIZE = 200
# Number of epochs to train the net
NUM_EPOCHS = 15


def gen_data(_A):
    X = np.zeros(shape=(N_BATCH, N_TIME_STEPS, _M))

    Z = np.random.uniform(low=-1, high=1, size=(1, _M))
    if np.random.random() <= 3./7:
        Z = Z * (Z > 0)
    if np.linalg.norm(Z) < 1e-6:
        Z[0][0] = 1
    
    tmp = np.dot(Z, _A)
    tmp /= np.linalg.norm(tmp)
    for j in range(N_TIME_STEPS):
        X[0, j, :] = tmp
    return Z, X


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

    network_output = lasagne.layers.get_output(l_out)
    MMM = T.eye(n=3,m=3) - l_rnn.W_hid_to_hid
    M_min_half = T.matrix('M_min_half')
    sliced_X = T.matrix("sliced_X")

    cost_plus = T.sum(
        (
            T.dot(
                M_min_half,
                T.transpose(sliced_X) - 
                T.dot(MMM, T.transpose(network_output))
            )
        )**2
    )
    cost_minus = - T.sum(
        (
            T.dot(
                M_min_half,
                T.transpose(sliced_X) - 
                T.dot(MMM, T.transpose(network_output))
            )
        )**2
    )

    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    all_params.pop(0) 
    all_params.pop(1)

    print "Computing updates ..."
    updates_plus = lasagne.updates.adam(cost_plus, all_params, LEARNING_RATE)
    updates_minus = lasagne.updates.adam(cost_minus, all_params, LEARNING_RATE)

    print "Compiling functions ..."
    train_plus = theano.function(
        [l_in.input_var, sliced_X, M_min_half],
        cost_plus, updates=updates_plus
    )
    train_minus = theano.function(
        [l_in.input_var, sliced_X, M_min_half],
        cost_minus, updates=updates_minus
    )

    compute_cost_plus = theano.function(
        [l_in.input_var, sliced_X, M_min_half], cost_plus
    )
    compute_cost_minus = theano.function(
        [l_in.input_var, sliced_X, M_min_half], cost_minus
    )

    result = theano.function([l_in.input_var],
                lasagne.layers.get_output(l_out))

    AA = np.random.uniform(low=-1, high=1, size=(3, 3))

    Z_fixed, X_fixed = gen_data(AA)
    X_sliced_fixed = X_fixed[:, 0, :]
    print("Z_fixed")
    print(Z_fixed)
    print("X_sliced_fixed")
    print(X_sliced_fixed)

    print("Initial cost:")
    MM_fixed = np.eye(3) - l_rnn.W_hid_to_hid.get_value()   #shape: (3, 3)
    M_half_fixed = sqrtm(MM_fixed)   #shape: (3, 3)
    M_minus_half_fixed = np.linalg.inv(M_half_fixed) #shape: (3, 3)
    if Z_fixed.min() >= 0:
        print compute_cost_plus(
            X_fixed.astype(theano.config.floatX),
            X_sliced_fixed.astype(theano.config.floatX),
            M_minus_half_fixed.astype(theano.config.floatX)
        )
    else:
        print compute_cost_minus(
            X_fixed.astype(theano.config.floatX),
            X_sliced_fixed.astype(theano.config.floatX),
            M_minus_half_fixed.astype(theano.config.floatX)
        )

    f = open('params.txt', 'w')   
    print "Training ..."
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                Z, X = gen_data(AA)
                X_sliced = X[:, 0, :]
                # Z = Z_fixed
                # X = X_fixed
                # X_sliced = X_sliced_fixed

                W = l_rnn.W_hid_to_hid.get_value()
                W_old = np.copy(W)
                MM = np.eye(3) - W  #shape: (3, 3)
                M_half = sqrtm(MM)   #shape: (3, 3)
                M_minus_half = np.linalg.inv(M_half)  #shape: (3, 3)

                if Z.min() >= 0:
                    train_plus(
                        X.astype(theano.config.floatX),
                        X_sliced.astype(theano.config.floatX),
                        M_minus_half.astype(theano.config.floatX) 
                    )
                else:
                    train_minus(
                        X.astype(theano.config.floatX),
                        X_sliced.astype(theano.config.floatX),
                        M_minus_half.astype(theano.config.floatX) 
                    )
                
                W_new = np.copy(l_rnn.W_hid_to_hid.get_value())
                delta_W = W_new - W_old
                delta_W = 1. / 2 * (delta_W + np.transpose(delta_W))

                S, V = LA.eig( W_old + delta_W )
                for i in range(S.size):
                    S[i] = max(-0.8, min(S[i], 0.8))
                W_new_final = np.dot(np.transpose(V), np.dot(np.diag(S), V))

                l_rnn.W_hid_to_hid.set_value(W_new_final)
                f.write("X_sliced: %s \nZ:        %s \n" % (X_sliced, Z))
                f.write("W_old:\n%s \n\n" % W_old)
                f.write("MM:\n%s\n\n" % MM)
                f.write("M_half:\n%s\n\n" % M_half)
                f.write("M_minus_half:\n%s\n\n" % M_minus_half)
                f.write("W_new:\n%s \n\n" % W_new)
                f.write("W_new - W_old:\n%s \n\n" % (W_new - W_old))
                f.write("delta_W:\n%s \n\n" % delta_W)
                f.write("W_new_final (all_params):\n%s \n\n" % all_params[0].get_value())
                if Z.min() >=0:
                    current_cost = compute_cost_plus(
                        X.astype(theano.config.floatX),
                        X_sliced.astype(theano.config.floatX),
                        M_minus_half.astype(theano.config.floatX)
                    )
                else:
                    current_cost = compute_cost_minus(
                        X.astype(theano.config.floatX),
                        X_sliced.astype(theano.config.floatX),
                        M_minus_half.astype(theano.config.floatX)
                    )
                f.write("Current cost: %s\n" % current_cost)
                f.write("%s \n" % "**********************************************")

            print "Cost:"
            MM = np.eye(3) - l_rnn.W_hid_to_hid.get_value()    #shape: (3, 3)
            M_half = sqrtm(MM)   #shape: (3, 3)
            M_minus_half = np.linalg.inv(M_half)  #shape: (3, 3)
            if Z_fixed.min() >=0:
                print compute_cost_plus(
                    X_fixed.astype(theano.config.floatX),
                    X_sliced_fixed.astype(theano.config.floatX),
                    M_minus_half.astype(theano.config.floatX)
                )
            else:
                print compute_cost_minus(
                    X_fixed.astype(theano.config.floatX),
                    X_sliced_fixed.astype(theano.config.floatX),
                    M_minus_half.astype(theano.config.floatX)
                )

    except KeyboardInterrupt:
        pass

    f.close()

    lst_cost_from_cone = []
    lst_cost_not_from_cone = []
    MM = np.eye(3) - l_rnn.W_hid_to_hid.get_value()    #shape: (3, 3)
    M_half = sqrtm(MM)   #shape: (3, 3)
    M_minus_half = np.linalg.inv(M_half)  #shape: (3, 3)

    for _ in range(100):
        X = np.zeros(shape=(N_BATCH, N_TIME_STEPS, _M))
        Z = np.random.uniform(low=0, high=1, size=(1, _M))
        # if np.random.random() <= 3./7:
        #     Z = Z * (Z > 0)
        if np.linalg.norm(Z) < 1e-6:
            Z[0][0] = 1
        tmp = np.dot(Z, AA)
        tmp /= np.linalg.norm(tmp)
        for j in range(N_TIME_STEPS):
            X[0, j, :] = tmp
        X_sliced = X[:, 0, :]

        lst_cost_from_cone.append(
            compute_cost_plus(
                X.astype(theano.config.floatX),
                X_sliced.astype(theano.config.floatX),
                M_minus_half.astype(theano.config.floatX)
            )
        )
    print("mean cost (from cone): %f" % np.mean(lst_cost_from_cone))

    for _ in range(100):
        X = np.zeros(shape=(N_BATCH, N_TIME_STEPS, _M))
        Z = np.random.uniform(low=-1, high=1, size=(1, _M))
        if Z.min() >= 0:
            k = np.random.randint(0, 3)
            Z[0][k] *= -1
        if np.linalg.norm(Z) < 1e-6:
            if Z[0][0] >= 0:
                Z[0][0] = 1
            elif Z[0][1] >=0 :
                Z[0][1] = 1
        tmp = np.dot(Z, AA)
        tmp /= np.linalg.norm(tmp)
        for j in range(N_TIME_STEPS):
            X[0, j, :] = tmp
        X_sliced = X[:, 0, :]

        lst_cost_not_from_cone.append(
            compute_cost_plus(
                X.astype(theano.config.floatX),
                X_sliced.astype(theano.config.floatX),
                M_minus_half.astype(theano.config.floatX)
            )
        )

    print("mean cost (not from cone): %f" % np.mean(lst_cost_not_from_cone))



if __name__ == '__main__':
    main()
