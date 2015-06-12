"""
TODO try to move these tests to the appropriate locations, and don't use
the lasagne wrapped classes
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano

import treeano
import treeano.lasagne
from treeano import UpdateDeltas
from treeano.lasagne.inits import GlorotUniformInit
from treeano.nodes import (InputNode,
                           SequentialNode,
                           IdentityNode,
                           HyperparameterNode)
from treeano.lasagne.nodes import (DenseNode,
                                   ReLUNode)

floatX = theano.config.floatX


def test_identity_network():
    input_node = InputNode("foo", shape=(3, 4, 5))
    network = input_node.build()
    fn = network.function(["foo"], ["foo"])
    x = np.random.rand(3, 4, 5).astype(floatX)
    assert np.allclose(fn(x), x)


def test_sequential_identity_network():
    nodes = [
        InputNode("foo", shape=(3, 4, 5)),
        IdentityNode("bar"),
    ]
    sequential = SequentialNode("choo", nodes)
    network = sequential.build()
    fn1 = network.function(["foo"], ["foo"])
    fn2 = network.function(["foo"], ["bar"])
    fn3 = network.function(["foo"], ["choo"])
    x = np.random.rand(3, 4, 5).astype(floatX)
    assert np.allclose(fn1(x), x)
    assert np.allclose(fn2(x), x)
    assert np.allclose(fn3(x), x)


def test_nested_sequential_network():
    current_node = InputNode("foo", shape=(3, 4, 5))
    for name in map(str, range(10)):
        current_node = SequentialNode("sequential" + name,
                                      [current_node,
                                       IdentityNode("identity" + name)])
    network = current_node.build()
    fn = network.function(["foo"], ["sequential9"])
    x = np.random.rand(3, 4, 5).astype(floatX)
    assert np.allclose(fn(x), x)


def test_toy_updater_node():

    class ToyUpdaterNode(treeano.NodeImpl):

        """
        example node to test compute_update_deltas
        """

        input_keys = ()

        def compute_output(self, network):
            shape = (2, 3, 4)
            state = network.create_variable(
                name="default",
                shape=shape,
                is_shared=True,
                tags=["state"],
            )
            init_value = np.arange(
                np.prod(shape)
            ).reshape(*shape).astype(floatX)
            state.value = init_value

        def new_update_deltas(self, network):
            return UpdateDeltas({
                network.get_variable("default").variable: 42
            })

    network = ToyUpdaterNode("a").build()
    fn1 = network.function([], ["a"])
    init_value = fn1()
    fn2 = network.function([], ["a"], include_updates=True)
    np.testing.assert_allclose(init_value[0],
                               fn2()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value[0] + 42,
                               fn2()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value[0] + 84,
                               fn1()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value[0] + 84,
                               fn1()[0],
                               rtol=1e-5,
                               atol=1e-8)


def test_hyperparameter_node():
    input_node = InputNode("a", shape=(3, 4, 5))
    hp_node = HyperparameterNode("b", input_node, foo=3, bar=2)
    network = hp_node.build()
    assert network["b"].find_hyperparameter(["foo"]) == 3
    assert network["a"].find_hyperparameter(["foo"]) == 3
    assert network["a"].find_hyperparameter(["choo", "foo"]) == 3
    assert network["a"].find_hyperparameter(["choo"], 32) == 32


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
    network = hp_node.build()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(floatX)
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
    network = hp_node.build()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(floatX)
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
    network = hp_node.build()
    fc_node = network["b"]
    W_value = fc_node.get_variable("W").value
    b_value = fc_node.get_variable("b").value
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
