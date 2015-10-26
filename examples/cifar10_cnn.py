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

fX = theano.config.floatX
BATCH_SIZE = 256
train, valid, test = canopy.sandbox.datasets.cifar10()

# based off of architecture from "Scalable Bayesian Optimization Using
# Deep Neural Networks" http://arxiv.org/abs/1502.05700
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(BATCH_SIZE, 3, 32, 32)),
         tn.DnnConv2DWithBiasNode("conv1", num_filters=96),
         tn.ReLUNode("relu1"),
         tn.DnnConv2DWithBiasNode("conv2", num_filters=96),
         tn.ReLUNode("relu2"),
         tn.MaxPool2DNode("mp1"),
         tn.DropoutNode("do1", dropout_probability=0.1),
         tn.DnnConv2DWithBiasNode("conv3", num_filters=192),
         tn.ReLUNode("relu3"),
         tn.DnnConv2DWithBiasNode("conv4", num_filters=192),
         tn.ReLUNode("relu4"),
         tn.DnnConv2DWithBiasNode("conv5", num_filters=192),
         tn.ReLUNode("relu5"),
         tn.MaxPool2DNode("mp2"),
         tn.DropoutNode("do2", dropout_probability=0.5),
         tn.DnnConv2DWithBiasNode("conv6", num_filters=192),
         tn.ReLUNode("relu6"),
         tn.DnnConv2DWithBiasNode("conv7",
                                  num_filters=192,
                                  filter_size=(1, 1)),
         tn.ReLUNode("relu7"),
         tn.DnnConv2DWithBiasNode("conv8",
                                  num_filters=10,
                                  filter_size=(1, 1)),
         tn.GlobalMeanPool2DNode("mean_pool"),
         tn.SoftmaxNode("pred"),
         ]),
    filter_size=(3, 3),
    conv_pad="same",
    pool_size=(3, 3),
    pool_stride=(2, 2),
    pool_pad=(1, 1),
    inits=[treeano.inits.OrthogonalInit()],
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
