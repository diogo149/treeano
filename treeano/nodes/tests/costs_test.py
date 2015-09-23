import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_aggregator_node_serialization():
    tn.check_serialization(tn.AggregatorNode("a"))


def test_elementwise_cost_node_serialization():
    tn.check_serialization(tn.ElementwiseCostNode(
        "foo",
        {"pred": tn.IdentityNode("foo"),
         "target": tn.IdentityNode("bar")}))


def test_total_cost_node_serialization():
    tn.check_serialization(tn.TotalCostNode(
        "foo",
        {"pred": tn.IdentityNode("foo"),
         "target": tn.IdentityNode("bar")}))


def test_auxilliary_cost_node_serialization():
    tn.check_serialization(tn.AuxiliaryCostNode(
        "foo",
        {"target": tn.IdentityNode("bar")}))


def test_total_cost_node():
    network = tn.TotalCostNode(
        "cost",
        {"pred": tn.InputNode("x", shape=(3, 4, 5)),
         "target": tn.InputNode("y", shape=(3, 4, 5))},
        cost_function=treeano.utils.squared_error).network()
    fn = network.function(["x", "y"], ["cost"])
    x = np.random.rand(3, 4, 5).astype(fX)
    y = np.random.rand(3, 4, 5).astype(fX)
    np.testing.assert_allclose(fn(x, y)[0],
                               ((x - y) ** 2).mean(),
                               rtol=1e-5)
    np.testing.assert_allclose(fn(x, x)[0],
                               0)
    np.testing.assert_allclose(fn(y, y)[0],
                               0)


def test_total_cost_node_with_weight():
    network = tn.TotalCostNode(
        "cost",
        {"pred": tn.InputNode("x", shape=(3, 4, 5)),
         "weight": tn.InputNode("w", shape=(3, 4, 5)),
         "target": tn.InputNode("y", shape=(3, 4, 5))},
        cost_function=treeano.utils.squared_error).network()
    fn = network.function(["x", "y", "w"], ["cost"])
    x = np.random.rand(3, 4, 5).astype(fX)
    w = np.random.rand(3, 4, 5).astype(fX)
    y = np.random.rand(3, 4, 5).astype(fX)
    np.testing.assert_allclose(fn(x, y, w)[0],
                               (((x - y) ** 2) * w).mean(),
                               rtol=1e-5)
    np.testing.assert_allclose(fn(x, x, w)[0],
                               0)
    np.testing.assert_allclose(fn(y, y, w)[0],
                               0)


def test_auxiliary_cost_node():
    network = tn.HyperparameterNode(
        "hp",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("x", shape=(3, 4, 5)),
             tn.AuxiliaryCostNode(
                 "cost1",
                 {"target": tn.InputNode("y1", shape=(3, 4, 5))}),
             tn.AddConstantNode("a1", value=2),
             tn.AuxiliaryCostNode(
                 "cost2",
                 {"target": tn.InputNode("y2", shape=(3, 4, 5))}),
             tn.MultiplyConstantNode("m1", value=2),
             tn.AuxiliaryCostNode(
                 "cost3",
                 {"target": tn.InputNode("y3", shape=(3, 4, 5))}),
             tn.ConstantNode("const", value=0),
             tn.InputElementwiseSumNode("cost")]
        ),
        cost_reference="cost",
        cost_function=treeano.utils.squared_error,
    ).network()
    fn = network.function(["x", "y1", "y2", "y3"], ["cost"])
    x = np.random.rand(3, 4, 5).astype(fX)
    ys = [np.random.rand(3, 4, 5).astype(fX) for _ in range(3)]

    def mse(x, y):
        return ((x - y) ** 2).mean()

    expected_output = (mse(x, ys[0])
                       + mse(x + 2, ys[1])
                       + mse(2 * (x + 2), ys[2]))
    np.testing.assert_allclose(fn(x, *ys)[0],
                               expected_output,
                               rtol=1e-5)
