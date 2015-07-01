import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import batch_normalization

fX = theano.config.floatX


def test_batch_normalization_node():
    network = tn.AdamNode(
        "adam",
        {"subtree": tn.SequentialNode(
            "seq",
            [tn.InputNode("x", shape=(None, 10)),
             batch_normalization.BatchNormalizationNode("bn"),
             tn.DenseNode("d", num_units=1), ]),
         "cost": tn.TotalCostNode("cost", {
             "target": tn.InputNode("y", shape=(None, 1)),
             "pred": tn.ReferenceNode("pred_ref", reference="d"),
         },
            cost_function=treeano.utils.squared_error)}).network()

    fn = network.function(["x", "y"], ["cost"], include_updates=True)

    x = 100 + 100 * np.random.randn(100, 10).astype(fX)
    y = np.random.randn(100, 1).astype(fX)
    prev_cost = fn(x, y)[0]
    for _ in range(3):
        cost = fn(x, y)[0]
        assert cost < prev_cost
        prev_cost = cost
