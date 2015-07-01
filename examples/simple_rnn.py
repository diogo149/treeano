from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX

# ################################## config ##################################

N_TRAIN = 5000
LAG = 20
LENGTH = 50
HIDDEN_STATE_SIZE = 10

# ############################### prepare data ###############################


def binary_toy_data(lag=1, length=20):
    inputs = np.random.randint(0, 2, length).astype(fX)
    outputs = np.array(lag * [0] + list(inputs), dtype=fX)[:length]
    return inputs, outputs


# ############################## prepare model ##############################

model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 1)),
         tn.recurrent.SimpleRecurrentNode(
             "srn",
             tn.TanhNode("nonlin"),
             batch_size=None,
             num_units=HIDDEN_STATE_SIZE),
         tn.scan.ScanNode(
             "scan",
             tn.DenseNode("fc", num_units=1)),
         tn.SigmoidNode("pred"),
         ]),
    inits=[treeano.inits.NormalWeightInit(0.01)],
    scan_axis=0
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.TotalCostNode("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None, 1))},
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
    inputs, outputs = binary_toy_data(lag=LAG, length=LENGTH)
    loss = train_fn(inputs.reshape(-1, 1), outputs.reshape(-1, 1))[0]
    if (i % (N_TRAIN // 100)) == 0:
        print(loss)
print("total_time: %s" % (time.time() - st))

inputs, outputs = binary_toy_data(lag=LAG, length=LENGTH)
pred = valid_fn(inputs.reshape(-1, 1))[0].flatten()
print(np.round(pred) == outputs)
