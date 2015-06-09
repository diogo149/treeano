import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from .. import utils

fX = theano.config.floatX


def test_dense_node_serialization():
    tn.check_serialization(tn.DenseNode("a"))
    tn.check_serialization(tn.DenseNode("a", num_units=100))


def test_dense_combine_node_serialization():
    tn.check_serialization(tn.DenseCombineNode("a", []))
    tn.check_serialization(tn.DenseCombineNode("a", [], num_units=100))


def test_dense_node():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("in", shape=(3, 4, 5)),
         tn.DenseNode("fc1", num_units=6),
         tn.DenseNode("fc2", num_units=7),
         tn.DenseNode("fc3", num_units=8)]
    ).build()
    x = np.random.randn(3, 4, 5).astype(fX)
    fn = network.function(["in"], ["fc3"])
    res = fn(x)[0]
    nt.assert_equal(res.shape, (3, 8))


def test_dense_combine_node():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("in", shape=(3, 4, 5)),
         tn.DenseCombineNode("fc1", [tn.IdentityNode("i1")], num_units=6),
         tn.DenseCombineNode("fc2", [tn.IdentityNode("i2")], num_units=7),
         tn.DenseCombineNode("fc3", [tn.IdentityNode("i3")], num_units=8)]
    ).build()
    x = np.random.randn(3, 4, 5).astype(fX)
    fn = network.function(["in"], ["fc3"])
    res = fn(x)[0]
    nt.assert_equal(res.shape, (3, 8))


def test_dense_node_and_dense_combine_node():
    # testing that dense node and dense combine node with identity child
    # return the same thing
    network1 = tn.HyperparameterNode(
        "hp",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("in", shape=(3, 4, 5)),
             tn.DenseNode("fc1", num_units=6),
             tn.DenseNode("fc2", num_units=7),
             tn.DenseNode("fc3", num_units=8)]
        ),
        shared_initializations=[utils.OnesInitialization()]
    ).build()
    network2 = tn.HyperparameterNode(
        "hp",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("in", shape=(3, 4, 5)),
             tn.DenseCombineNode("fc1", [tn.IdentityNode("i1")], num_units=6),
             tn.DenseCombineNode("fc2", [tn.IdentityNode("i2")], num_units=7),
             tn.DenseCombineNode("fc3", [tn.IdentityNode("i3")], num_units=8)]
        ),
        shared_initializations=[utils.OnesInitialization()]
    ).build()
    x = np.random.randn(3, 4, 5).astype(fX)
    fn1 = network1.function(["in"], ["fc3"])
    fn2 = network2.function(["in"], ["fc3"])
    np.testing.assert_allclose(fn1(x), fn2(x))
