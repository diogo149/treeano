import numpy as np
import theano
import theano.tensor as T

fX = theano.config.floatX

LAG = 20
LENGTH = 50
N_TRAIN = 5000
HIDDEN_STATE_SIZE = 10


def binary_toy_data(lag=1, length=20):
    inputs = np.random.randint(0, 2, length).astype(fX)
    outputs = np.array(lag * [0] + list(inputs), dtype=fX)[:length]
    return inputs, outputs


W_x = theano.shared(
    (0.1 * np.random.randn(1, HIDDEN_STATE_SIZE)).astype(fX))
W_h = theano.shared(
    (0.1 * np.random.randn(HIDDEN_STATE_SIZE,
                           HIDDEN_STATE_SIZE)).astype(fX))
W_y = theano.shared(
    (0.1 * np.random.randn(HIDDEN_STATE_SIZE, 1)).astype(fX))

b_h = theano.shared(np.zeros((HIDDEN_STATE_SIZE,), dtype=fX))
b_y = theano.shared(np.zeros((1,), dtype=fX))


X = T.matrix("X")
Y = T.matrix("Y")


def step(x, h):
    new_h = T.tanh(T.dot(x, W_x) + T.dot(h, W_h) + b_h)
    new_y = T.nnet.sigmoid(T.dot(new_h, W_y) + b_y)
    return new_h, new_y


results, updates = theano.scan(
    fn=step,
    sequences=[X],
    outputs_info=[T.patternbroadcast(T.zeros((HIDDEN_STATE_SIZE)),
                                     (False,)), None],
)
ys = results[1]

loss = T.mean((ys - Y) ** 2)
params = [W_x, W_h, W_y, b_h, b_y]
grads = T.grad(loss, params)
updates = []
for param, grad in zip(params, grads):
    updates.append((param, param - grad * 0.1))

train_fn = theano.function([X, Y], loss, updates=updates)
valid_fn = theano.function([X], ys)


import time
st = time.time()
for i in range(N_TRAIN):
    inputs, outputs = binary_toy_data(lag=LAG, length=LENGTH)
    loss = train_fn(inputs.reshape(-1, 1), outputs.reshape(-1, 1))
    if (i % (N_TRAIN // 100)) == 0:
        print(loss)
print "total_time: %s" % (time.time() - st)

inputs, outputs = binary_toy_data(lag=LAG, length=LENGTH)
preds = valid_fn(inputs.reshape(-1, 1)).flatten()
print(np.round(preds) == outputs)
