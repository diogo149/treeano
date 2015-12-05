from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import itertools
import pprint
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

if 0:
    concat_node = tn.IdentityNode
else:
    from treeano.sandbox.nodes import activation_transformation
    concat_node = activation_transformation.ConcatenateNegationNode

num_filters_1 = 96
num_filters_2 = 192

# based off of architecture from "Scalable Bayesian Optimization Using
# Deep Neural Networks" http://arxiv.org/abs/1502.05700
model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 3, 32, 32)),
         tn.DnnConv2DWithBiasNode("conv1", num_filters=num_filters_1),
         concat_node("concat1"),
         tn.ReLUNode("relu1"),
         tn.DropoutNode("do1"),
         tn.DnnConv2DWithBiasNode("conv2", num_filters=num_filters_1),
         concat_node("concat2"),
         tn.ReLUNode("relu2"),
         tn.DropoutNode("do2"),
         tn.MaxPool2DNode("mp1"),
         tn.DnnConv2DWithBiasNode("conv3", num_filters=num_filters_2),
         concat_node("concat3"),
         tn.ReLUNode("relu3"),
         tn.DropoutNode("do3"),
         tn.DnnConv2DWithBiasNode("conv4", num_filters=num_filters_2),
         concat_node("concat4"),
         tn.ReLUNode("relu4"),
         tn.DropoutNode("do4"),
         tn.DnnConv2DWithBiasNode("conv5", num_filters=num_filters_2),
         concat_node("concat5"),
         tn.ReLUNode("relu5"),
         tn.DropoutNode("do5"),
         tn.MaxPool2DNode("mp2"),
         tn.DnnConv2DWithBiasNode("conv6", num_filters=num_filters_2),
         concat_node("concat6"),
         tn.ReLUNode("relu6"),
         tn.DropoutNode("do6"),
         tn.DnnConv2DWithBiasNode("conv7",
                                  num_filters=num_filters_2,
                                  filter_size=(1, 1)),
         concat_node("concat7"),
         tn.ReLUNode("relu7"),
         tn.DropoutNode("do7"),
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
    dropout_probability=0.3,
    inits=[treeano.inits.OrthogonalInit()],
)

update_node = tn.AdamNode
cost_node = tn.TotalCostNode
model = tn.HyperparameterNode(
    "with_updates",
    update_node(
        "adam",
        {"subtree": model,
         "cost": cost_node("cost", {
             "pred": tn.ReferenceNode("pred_ref", reference="model"),
             "target": tn.InputNode("y", shape=(None,), dtype="int32")},
         )},
        learning_rate=2e-4),
    cost_function=treeano.utils.categorical_crossentropy_i32,
)

network = model.network()
network.build()  # build eagerly to share weights
print(network.root_node)

valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="valid_time"),
     canopy.handlers.evaluate_monitoring_variables(fmt="valid_%s"),
     canopy.handlers.override_hyperparameters(deterministic=True),
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
     canopy.handlers.evaluate_monitoring_variables(fmt="train_%s"),
     canopy.handlers.batch_pad(BATCH_SIZE, keys=["x", "y"]),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"train_cost": "cost"},
    include_updates=True)

# ################################# training #############################

BEST_ACC = [0]
LOGS = []


def callback(log):
    LOGS.append(log)
    pprint.pprint(log)
    BEST_ACC[0] = max(BEST_ACC[0], log["valid_accuracy"])
    print("best: %0.4g" % BEST_ACC[0])


def train_generator():
    x = train["x"]
    y = train["y"]
    chunk_size = len(train["x"])
    while True:
        x_chunk = []
        y_chunk = []
        for _ in range(chunk_size):
            idx = np.random.randint(len(x))
            tmp_x = x[idx]
            # flip axes
            if np.random.rand() < 0.5:
                tmp_x = tmp_x[:, :, ::-1]
            x_chunk.append(tmp_x)
            y_chunk.append(y[idx])
        yield {"x": np.array(x_chunk), "y": np.array(y_chunk)}

print("Starting training...")
canopy.evaluate_until(fn=train_fn,
                      gen=train_generator(),
                      # max_iters=200,
                      # HACK
                      max_iters=10000,
                      callback=callback)
