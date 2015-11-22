"""
TODO try to move these tests to the appropriate locations
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano

import treeano
import treeano.lasagne
from treeano import UpdateDeltas
from treeano.nodes import (InputNode,
                           SequentialNode,
                           IdentityNode,
                           HyperparameterNode)

fX = theano.config.floatX


def test_identity_network():
    input_node = InputNode("foo", shape=(3, 4, 5))
    network = input_node.network()
    fn = network.function(["foo"], ["foo"])
    x = np.random.rand(3, 4, 5).astype(fX)
    assert np.allclose(fn(x), x)


def test_sequential_identity_network():
    nodes = [
        InputNode("foo", shape=(3, 4, 5)),
        IdentityNode("bar"),
    ]
    sequential = SequentialNode("choo", nodes)
    network = sequential.network()
    fn1 = network.function(["foo"], ["foo"])
    fn2 = network.function(["foo"], ["bar"])
    fn3 = network.function(["foo"], ["choo"])
    x = np.random.rand(3, 4, 5).astype(fX)
    assert np.allclose(fn1(x), x)
    assert np.allclose(fn2(x), x)
    assert np.allclose(fn3(x), x)


def test_nested_sequential_network():
    current_node = InputNode("foo", shape=(3, 4, 5))
    for name in map(str, range(10)):
        current_node = SequentialNode("sequential" + name,
                                      [current_node,
                                       IdentityNode("identity" + name)])
    network = current_node.network()
    fn = network.function(["foo"], ["sequential9"])
    x = np.random.rand(3, 4, 5).astype(fX)
    assert np.allclose(fn(x), x)


def test_toy_updater_node():

    class ToyUpdaterNode(treeano.NodeImpl):

        """
        example node to test compute_update_deltas
        """

        input_keys = ()

        def compute_output(self, network):
            shape = (2, 3, 4)
            state = network.create_vw(
                name="default",
                shape=shape,
                is_shared=True,
                tags=["state"],
                inits=[],
            )
            init_value = np.arange(
                np.prod(shape)
            ).reshape(*shape).astype(fX)
            state.value = init_value

        def new_update_deltas(self, network):
            return UpdateDeltas({
                network.get_vw("default").variable: 42
            })

    network = ToyUpdaterNode("a").network()
    fn1 = network.function([], ["a"])
    # cast to np.array, since it could be a CudaNdarray if on gpu
    init_value = np.array(fn1()[0])
    fn2 = network.function([], ["a"], include_updates=True)
    np.testing.assert_allclose(init_value,
                               fn2()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value + 42,
                               fn2()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value + 84,
                               fn1()[0],
                               rtol=1e-5,
                               atol=1e-8)
    np.testing.assert_allclose(init_value + 84,
                               fn1()[0],
                               rtol=1e-5,
                               atol=1e-8)


def test_hyperparameter_node():
    input_node = InputNode("a", shape=(3, 4, 5))
    hp_node = HyperparameterNode("b", input_node, foo=3, bar=2)
    network = hp_node.network()
    assert network["b"].find_hyperparameter(["foo"]) == 3
    assert network["a"].find_hyperparameter(["foo"]) == 3
    assert network["a"].find_hyperparameter(["choo", "foo"]) == 3
    assert network["a"].find_hyperparameter(["choo"], 32) == 32
