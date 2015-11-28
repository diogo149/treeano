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
from treeano.sandbox.nodes import spatial_transformer as st
from treeano.sandbox.nodes import batch_normalization as bn
from treeano.sandbox.nodes import spatial_attention
import canopy
import canopy.sandbox.datasets

fX = theano.config.floatX

UPDATE_SCALE_FACTOR = 1.0
MAX_ITERS = 100
BATCH_SIZE = 500

train, valid, _ = canopy.sandbox.datasets.cluttered_mnist()

# ############################## prepare model ##############################

localization_network = tn.HyperparameterNode(
    "loc",
    tn.SequentialNode(
        "loc_seq",
        [tn.DnnMaxPoolNode("loc_pool1"),
         tn.DnnConv2DWithBiasNode("loc_conv1"),
         tn.DnnMaxPoolNode("loc_pool2"),
         bn.NoScaleBatchNormalizationNode("loc_bn1"),
         tn.ReLUNode("loc_relu1"),
         tn.DnnConv2DWithBiasNode("loc_conv2"),
         bn.SimpleBatchNormalizationNode("loc_bn2"),
         tn.SpatialSoftmaxNode("loc_spatial_softmax"),
         spatial_attention.SpatialFeaturePointNode("loc_feature_point"),
         tn.DenseNode("loc_fc1", num_units=50),
         bn.NoScaleBatchNormalizationNode("loc_bn3"),
         tn.ReLUNode("loc_relu3"),
         tn.DenseNode("loc_fc2",
                      num_units=6,
                      inits=[treeano.inits.NormalWeightInit(std=0.001)])]),
    num_filters=20,
    filter_size=(5, 5),
    pool_size=(2, 2),
)

st_node = st.AffineSpatialTransformerNode(
    "st",
    localization_network,
    output_shape=(20, 20))

model = tn.HyperparameterNode(
    "model",
    tn.SequentialNode(
        "seq",
        [tn.InputNode("x", shape=(None, 1, 60, 60)),
         # scaling the updates of the spatial transformer
         # seems to be very helpful, to allow the clasification
         # net to learn what to look for, before prematurely
         # looking
         tn.UpdateScaleNode(
             "st_update_scale",
             st_node,
             update_scale_factor=UPDATE_SCALE_FACTOR),
         tn.Conv2DWithBiasNode("conv1"),
         tn.MaxPool2DNode("mp1"),
         bn.NoScaleBatchNormalizationNode("bn1"),
         tn.ReLUNode("relu1"),
         tn.Conv2DWithBiasNode("conv2"),
         tn.MaxPool2DNode("mp2"),
         bn.NoScaleBatchNormalizationNode("bn2"),
         tn.ReLUNode("relu2"),
         tn.GaussianDropoutNode("do1"),
         tn.DenseNode("fc1"),
         bn.NoScaleBatchNormalizationNode("bn3"),
         tn.ReLUNode("relu3"),
         tn.DenseNode("fc2", num_units=10),
         tn.SoftmaxNode("pred"),
         ]),
    num_filters=32,
    filter_size=(3, 3),
    pool_size=(2, 2),
    num_units=256,
    dropout_probability=0.5,
    inits=[treeano.inits.HeUniformInit()],
    bn_update_moving_stats=True,
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
    learning_rate=2e-3,
)
network = with_updates.network()
network.build()  # build eagerly to share weights


valid_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="valid_time"),
     canopy.handlers.override_hyperparameters(deterministic=True),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"valid_cost": "cost", "pred": "pred"})


def validate(in_dict, result_dict):
    valid_out = valid_fn(valid)
    probabilities = valid_out.pop("pred")
    predicted_classes = np.argmax(probabilities, axis=1)
    result_dict["valid_accuracy"] = sklearn.metrics.accuracy_score(
        valid["y"], predicted_classes)
    result_dict.update(valid_out)

train_fn = canopy.handled_fn(
    network,
    [canopy.handlers.time_call(key="total_time"),
     canopy.handlers.call_after_every(1, validate),
     canopy.handlers.time_call(key="train_time"),
     canopy.handlers.chunk_variables(batch_size=BATCH_SIZE,
                                     variables=["x", "y"])],
    {"x": "x", "y": "y"},
    {"train_cost": "cost"},
    include_updates=True)


def callback(results_dict):
    print("{_iter:3d}: "
          "train_cost: {train_cost:0.3f} "
          "valid_cost: {valid_cost:0.3f} "
          "valid_accuracy: {valid_accuracy:0.3f}".format(**results_dict))

print("Starting training...")
canopy.evaluate_until(fn=train_fn,
                      gen=itertools.repeat(train),
                      max_iters=MAX_ITERS,
                      callback=callback)
