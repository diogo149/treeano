from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import recurrent_hc

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

model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, None, 1)),
         recurrent_hc.GRUNode("gru1"),
         tn.LinearMappingNode("y_linear", output_dim=1),
         tn.AddBiasNode("y_bias", broadcastable_axes=(0, 1)),
         tn.SigmoidNode("sigmoid"),
         ]),
    inits=[treeano.inits.OrthogonalInit()],
    num_units=HIDDEN_STATE_SIZE,
    learn_init=True,
    grad_clip=1,
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.TotalCostNode("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None, None, 1))},
         )}),
    cost_function=treeano.utils.squared_error,
)
network = with_updates.network()

train_fn = network.function(["x", "y"], ["cost"], include_updates=True)
valid_fn = network.function(["x"], ["model"])


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
