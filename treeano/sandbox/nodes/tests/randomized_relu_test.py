from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import randomized_relu as rrelu


fX = theano.config.floatX


def test_rrelu_node_serialization():
    tn.check_serialization(rrelu.RandomizedReLUNode("a"))


def test_rrelu_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 4)),
         rrelu.RandomizedReLUNode("p", deterministic=True)]).network()

    fn = network.function(["i"], ["p"])
    x = np.array([[-1.0, -0.2, 0.2, 1.0]], dtype=fX)
    ans = np.array([[-1.0 * 2 / 11, -0.2 * 2 / 11, 0.2, 1.0]], dtype=fX)
    np.testing.assert_allclose(fn(x)[0],
                               ans,
                               rtol=1e-5)


def test_rrelu_node2():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(100, 100)),
         rrelu.RandomizedReLUNode("p")]).network()

    fn = network.function(["i"], ["p"])
    x = -np.random.rand(100, 100).astype(fX)
    res = fn(x)[0]
    assert res.min() > -1 / 3.
    assert res.max() < 0
