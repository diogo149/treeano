import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
from treeano import nodes

floatX = theano.config.floatX


def test_dense_node_serialization():
    nodes.check_serialization(nodes.DenseNode("a",
                                              nodes.toy.AddConstantNode("b")))
    nodes.check_serialization(nodes.DenseNode("a",
                                              nodes.toy.AddConstantNode("b"),
                                              num_units=100))


def test_dense_node():
    network = nodes.SequentialNode(
        "seq",
        [nodes.InputNode("in", shape=(3, 4, 5)),
         nodes.DenseNode("fc1", nodes.IdentityNode("i1"), num_units=6),
         nodes.DenseNode("fc2", nodes.IdentityNode("i2"), num_units=7),
         nodes.DenseNode("fc3", nodes.IdentityNode("i3"), num_units=8)]
    ).build()
    x = np.random.randn(3, 4, 5).astype(floatX)
    fn = network.function(["in"], ["fc3"])
    res = fn(x)[0]
    nt.assert_equal(res.shape, (3, 8))
