from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import itertools
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy
import canopy.sandbox.datasets

from treeano.sandbox.nodes import batch_normalization as bn
from treeano.sandbox.nodes import resnet

fX = theano.config.floatX
BATCH_SIZE = 256
train, valid, test = canopy.sandbox.datasets.cifar10()

groups = 3
blocks_per_group = 5
num_layers = 2
num_filters = 16

nodes = [
    tn.InputNode("x", shape=(None, 3, 32, 32)),
    tn.Conv2DNode("conv1", num_filters=num_filters),
    bn.BatchNormalizationNode("bn1"),
    tn.ReLUNode("relu1"),
]

for group in range(groups):
    for block in range(blocks_per_group):
        if group != 0 and block == 0:
            num_filters *= 2
            nodes.append(resnet.residual_block_conv_2d(
                "resblock_%d_%d" % (group, block),
                num_filters=num_filters,
                num_layers=num_layers,
                increase_dim="projection"))
        else:
            nodes.append(resnet.residual_block_conv_2d(
                "resblock_%d_%d" % (group, block),
                num_filters=num_filters,
                num_layers=num_layers))

nodes += [
    tn.GlobalMeanPool2DNode("global_pool"),
    tn.DenseNode("logit", num_units=10),
    tn.SoftmaxNode("pred"),
]

model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode("seq", nodes),
    filter_size=(3, 3),
    inits=[treeano.inits.OrthogonalInit()],
    pad="same",
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.TotalCostNode("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None,), dtype="int32")},
         )}),
    cost_function=treeano.utils.categorical_crossentropy_i32,
)
network = with_updates.network()
network.build()  # build eagerly to share weights

valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="valid_time"),
     canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.batch_pad(BATCH_SIZE, keys=["x", "y"]),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"valid_cost": "cost", "pred": "pred"})


def validate(in_dict, results_dict):
    valid_out = valid_fn(valid)
    valid_y = valid["y"]
    probabilities = valid_out.pop("pred")[:len(valid_y)]
    predicted_classes = np.argmax(probabilities, axis=1)
    valid_out["valid_accuracy"] = (valid_y == predicted_classes).mean()
    results_dict.update(valid_out)

train_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="total_time"),
     canopy.handlers.call_after_every(1, validate),
     canopy.handlers.time_call(key="train_time"),
     canopy.handlers.batch_pad(BATCH_SIZE, keys=["x", "y"]),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"train_cost": "cost"},
    include_updates=True)


# ################################# training #################################

print("Starting training...")
canopy.evaluate_until(fn=train_fn,
                      gen=itertools.repeat(train),
                      max_iters=200)
