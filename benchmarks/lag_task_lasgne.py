from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import lasagne

fX = theano.config.floatX

# ################################## config ##################################

N_TRAIN = 1000
LAG = 10
LENGTH = 50
HIDDEN_STATE_SIZE = 10
BATCH_SIZE = 64

# ############################### prepare data ###############################


def binary_toy_data(lag=1, length=20):
    inputs = np.random.randint(0, 2, length).astype(fX)
    outputs = np.array(lag * [0] + list(inputs), dtype=fX)[:length]
    return inputs, outputs


def minibatch(lag, length, batch_size):
    inputs = []
    outputs = []
    for _ in range(batch_size):
        i, o = binary_toy_data(lag, length)
        inputs.append(i)
        outputs.append(o)
    return np.array(inputs)[..., np.newaxis], np.array(outputs)[..., np.newaxis]


# ############################## prepare model ##############################

l = lasagne.layers.InputLayer(shape=(None, None, 1))
l = lasagne.layers.LSTMLayer(l,
                             num_units=HIDDEN_STATE_SIZE,
                             grad_clipping=1,
                             learn_init=True)
l = lasagne.layers.ReshapeLayer(l, shape=(-1, HIDDEN_STATE_SIZE))
l = lasagne.layers.DenseLayer(l,
                              num_units=1,
                              nonlinearity=lasagne.nonlinearities.sigmoid)

in_var = T.tensor3()
targets = T.tensor3()
outputs = lasagne.layers.get_output(l, in_var).reshape(in_var.shape)
loss = T.mean((targets - outputs) ** 2)
all_params = lasagne.layers.get_all_params(l)
updates = lasagne.updates.adam(loss, all_params)

train_fn = theano.function([in_var, targets], [loss], updates=updates)
valid_fn = theano.function([in_var], [outputs])


# ################################# training #################################

print("Starting training...")

import time
st = time.time()
for i in range(N_TRAIN):
    inputs, outputs = minibatch(lag=LAG, length=LENGTH, batch_size=BATCH_SIZE)
    loss = train_fn(inputs, outputs)[0]
    print(loss)
print("total_time: %s" % (time.time() - st))

inputs, outputs = minibatch(lag=LAG, length=LENGTH, batch_size=BATCH_SIZE)
pred = valid_fn(inputs)[0]
pred_accuracies = (np.round(pred) == outputs).mean(axis=0)[LAG:]
print(pred_accuracies)
print(pred_accuracies.mean())
