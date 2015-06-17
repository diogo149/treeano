import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import anrat

fX = theano.config.floatX


def test_anrat_node():
    network = tn.AdamNode(
        "adam",
        {"subtree": tn.InputNode("x", shape=(None, 1)),
         "cost": anrat.ANRATNode("cost", {
             "target": tn.InputNode("y", shape=(None, 1)),
             "pred": tn.ReferenceNode("pred_ref", reference="x"),
         })}).network()

    fn = network.function(["x", "y"], ["cost"], include_updates=True)

    for x_raw, y_raw in [(3.4, 2),
                         (4.2, 4.2)]:
        x = np.array([[x_raw]], dtype=fX)
        y = np.array([[y_raw]], dtype=fX)
        prev_cost = fn(x, y)[0]
        for _ in range(3):
            cost = fn(x, y)[0]
            assert cost < prev_cost
            prev_cost = cost
