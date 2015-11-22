"""
TODO try to move these tests to the appropriate locations
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano

import treeano
import treeano.lasagne
from treeano.lasagne.inits import GlorotUniformInit
from treeano.nodes import (InputNode,
                           SequentialNode,
                           HyperparameterNode)
from treeano.lasagne.nodes import (DenseNode,
                                   ReLUNode)

fX = theano.config.floatX


def test_dense_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", shape=(3, 4, 5)),
        DenseNode("b"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode(
        "d",
        sequential,
        num_units=14,
        inits=[treeano.inits.ConstantInit(1)])
    network = hp_node.network()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(fX)
    res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
    np.testing.assert_allclose(fn(x)[0],
                               res,
                               rtol=1e-5,
                               atol=1e-8)


def test_fully_connected_and_relu_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", shape=(3, 4, 5)),
        DenseNode("b"),
        ReLUNode("e"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode(
        "d",
        sequential,
        num_units=14,
        inits=[treeano.inits.ConstantInit(1)])
    network = hp_node.network()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(fX)
    res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
    np.testing.assert_allclose(fn(x)[0],
                               np.clip(res, 0, np.inf),
                               rtol=1e-5,
                               atol=1e-8)


def test_glorot_uniform_initialization():
    np.random.seed(42)
    nodes = [
        InputNode("a", shape=(3, 4, 5)),
        DenseNode("b"),
        ReLUNode("e"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=1000,
                                 inits=[GlorotUniformInit()])
    network = hp_node.network()
    fc_node = network["b"]
    W_value = fc_node.get_vw("W").value
    b_value = fc_node.get_vw("b").value
    np.testing.assert_allclose(0,
                               W_value.mean(),
                               atol=1e-2)
    np.testing.assert_allclose(np.sqrt(2.0 / (20 + 1000)),
                               W_value.std(),
                               atol=1e-2)
    np.testing.assert_allclose(np.zeros(1000),
                               b_value,
                               rtol=1e-5,
                               atol=1e-8)
