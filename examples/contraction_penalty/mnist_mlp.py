from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import itertools
import numpy as np
import sklearn.datasets
import sklearn.cross_validation
import sklearn.metrics
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy
from treeano.sandbox.nodes import contraction_penalty as cp

fX = theano.config.floatX

# ############################### prepare data ###############################

mnist = sklearn.datasets.fetch_mldata('MNIST original')
# theano has a constant float type that it uses (float32 for GPU)
# also rescaling to [0, 1] instead of [0, 255]
X = mnist['data'].astype(fX) / 255.0
y = mnist['target'].astype("int32")
X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
    X, y, random_state=42)
in_train = {"x": X_train, "y": y_train}
in_valid = {"x": X_valid, "y": y_valid}

# ############################## prepare model ##############################

model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 28 * 28)),
         cp.AuxiliaryContractionPenaltyNode(
             "cp1",
             tn.SequentialNode(
                 "cp_seq1",
                 [tn.DenseNode("fc1"),
                  # the cost has nan's when using ReLU's
                  # TODO look into why
                  tn.AbsNode("abs1")]),
             cost_weight=1e1),
         # the cost has nan's when this is enabled
         # TODO look into why
         # tn.DropoutNode("do1"),
         cp.AuxiliaryContractionPenaltyNode(
             "cp2",
             tn.SequentialNode(
                 "cp_seq2",
                 [tn.DenseNode("fc2"),
                  # the cost has nan's when using ReLU's
                  # TODO look into why
                  tn.AbsNode("abs2")]),
             cost_weight=1e1),
         tn.DropoutNode("do2"),
         tn.DenseNode("fc3", num_units=10),
         tn.SoftmaxNode("pred"),
         tn.TotalCostNode(
             "cost",
             {"pred": tn.IdentityNode("pred_id"),
              "target": tn.InputNode("y", shape=(None,), dtype="int32")},
             cost_function=treeano.utils.categorical_crossentropy_i32),
         tn.InputElementwiseSumNode("total_cost")]),
    num_units=32,
    cost_reference="total_cost",
    dropout_probability=0.5,
    inits=[treeano.inits.XavierNormalInit()],
)

with_updates = tn.HyperparameterNode(
    "with_updates",
    tn.AdamNode(
        "adam",
        {"subtree": model,
         "cost": tn.ReferenceNode("cost_ref", reference="total_cost")}),
)
network = with_updates.network()
network.build()  # build eagerly to share weights

BATCH_SIZE = 500

valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="valid_time"),
     canopy.handlers.override_hyperparameters(dropout_probability=0),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"total_cost": "total_cost", "pred": "pred"})


def validate(in_dict, results_dict):
    valid_out = valid_fn(in_valid)
    probabilities = valid_out["pred"]
    predicted_classes = np.argmax(probabilities, axis=1)
    results_dict["valid_cost"] = valid_out["total_cost"]
    results_dict["valid_time"] = valid_out["valid_time"]
    results_dict["valid_accuracy"] = sklearn.metrics.accuracy_score(
        y_valid, predicted_classes)

train_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="total_time"),
     canopy.handlers.call_after_every(1, validate),
     canopy.handlers.time_call(key="train_time"),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"train_cost": "total_cost",
     "train_cp_cost1": "cp1_sendto",
     "train_cp_cost2": "cp2_sendto",
     "train_classification_cost": "cost"},
    include_updates=True)


# ################################# training #################################

print("Starting training...")
canopy.evaluate_until(fn=train_fn,
                      gen=itertools.repeat(in_train),
                      max_iters=25)
