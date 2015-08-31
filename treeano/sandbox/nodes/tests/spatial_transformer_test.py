from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import spatial_transformer

fX = theano.config.floatX


def test_affine_spatial_transformer_node_build():
    localization_network = tn.HyperparameterNode(
        "loc",
        tn.SequentialNode(
            "loc_seq",
            [tn.DenseNode("loc_fc1", num_units=50),
             tn.ReLUNode("loc_relu3"),
             tn.DenseNode("loc_fc2",
                          num_units=6,
                          inits=[treeano.inits.ZeroInit()])]),
        num_filters=32,
        filter_size=(5, 5),
        pool_size=(2, 2),
    )

    model = tn.HyperparameterNode(
        "model",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("x", shape=(None, 1, 60, 60)),
             spatial_transformer.AffineSpatialTransformerNode(
                 "st",
                 localization_network,
                 output_shape=(20, 20)),
             tn.DenseNode("fc1"),
             tn.ReLUNode("relu1"),
             tn.DropoutNode("do1"),
             tn.DenseNode("fc2", num_units=10),
             tn.SoftmaxNode("pred"),
             ]),
        num_filters=32,
        filter_size=(3, 3),
        pool_size=(2, 2),
        num_units=256,
        dropout_probability=0.5,
        inits=[treeano.inits.HeNormalInit()],
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
