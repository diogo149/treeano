"""
some utilities for testing nodes
"""
import json

import numpy as np
import theano

from .. import core
from . import simple
from . import containers
from . import costs
from . import composite
from . import activations

floatX = theano.config.floatX


def check_serialization(node):
    import nose.tools as nt
    # test __eq__
    nt.assert_equal(
        node,
        node)
    # test serialization
    nt.assert_equal(
        node,
        core.node_from_data(core.node_to_data(node)))
    # test that serialization is json-serializable
    nt.assert_equal(
        node,
        core.node_from_data(json.loads(json.dumps(core.node_to_data(node)))))


def check_updates_node(updates_node_cls, **hyperparameters):
    import nose.tools as nt
    np.random.seed(42)
    network = simple.HyperparameterNode(
        "g",
        updates_node_cls(
            "updates",
            {"subtree": containers.SequentialNode("seq", [
                simple.InputNode("input", shape=(3, 4, 5)),
                composite.DenseNode("b"),
                activations.ReLUNode("c")]),
             "cost": costs.PredictionCostNode("cost", {
                 "preds": simple.ReferenceNode("preds_ref", reference="seq"),
                 "target": simple.InputNode("target", shape=(3, 14))})
             },
            **hyperparameters),
        num_units=14,
        loss_function=lambda preds, y_true: (preds - y_true) ** 2,
        cost_reference="cost",
    ).network()
    fn = network.function(["input", "target"], ["cost"])
    fn2 = network.function(["input", "target"],
                           ["cost"],
                           include_updates=True)
    x = np.random.randn(3, 4, 5).astype(floatX)
    y = np.random.randn(3, 14).astype(floatX)
    initial_cost = fn(x, y)
    next_cost = fn(x, y)
    np.testing.assert_allclose(initial_cost,
                               next_cost,
                               rtol=1e-5,
                               atol=1e-8)
    prev_cost = fn2(x, y)
    for _ in range(10):
        current_cost = fn2(x, y)
        print(current_cost)
        nt.assert_greater(prev_cost, current_cost)
        prev_cost = current_cost
