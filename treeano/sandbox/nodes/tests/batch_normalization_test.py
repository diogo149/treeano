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


def test_simple_batch_normalization_node():
    def test_network_with_shape(shape):
        network = tn.SequentialNode(
            "seq",
            [tn.InputNode("x", shape=shape),
             batch_normalization.SimpleBatchNormalizationNode("bn")]
        ).network()
        fn = network.function(["x"], ["seq"])
        x = (100 * np.random.randn(*shape) + 3).astype(fX)
        axis = tuple([i for i in range(len(shape)) if i != 1])
        mean = x.mean(axis=axis, keepdims=True)
        std = np.sqrt(x.var(axis=axis, keepdims=True) + 1e-8)
        ans = (x - mean) / std
        res = fn(x)[0]
        np.testing.assert_allclose(ans, res, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.zeros(shape[1]),
                                   ans.mean(axis=axis),
                                   atol=1e-6)

    test_network_with_shape((12, 10))
    test_network_with_shape((13, 10, 24))
    test_network_with_shape((14, 10, 5, 3))


# FIXME test copy-pasted from above
def test_no_scale_batch_normalization_node():
    def test_network_with_shape(shape):
        network = tn.SequentialNode(
            "seq",
            [tn.InputNode("x", shape=shape),
             batch_normalization.NoScaleBatchNormalizationNode("bn")]
        ).network()
        fn = network.function(["x"], ["seq"])
        x = (100 * np.random.randn(*shape) + 3).astype(fX)
        axis = tuple([i for i in range(len(shape)) if i != 1])
        mean = x.mean(axis=axis, keepdims=True)
        std = np.sqrt(x.var(axis=axis, keepdims=True) + 1e-8)
        ans = (x - mean) / std
        res = fn(x)[0]
        np.testing.assert_allclose(ans, res, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.zeros(shape[1]),
                                   ans.mean(axis=axis),
                                   atol=1e-6)

    test_network_with_shape((12, 10))
    test_network_with_shape((13, 10, 24))
    test_network_with_shape((14, 10, 5, 3))
