from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import fields
import theano
import lasagne


import treeano
from treeano import Node, UpdateDeltas
from treeano.initialization import GlorotUniform
from treeano.node import (InputNode,
                          SequentialNode,
                          IdentityNode,
                          HyperparameterNode,
                          CostNode,
                          ContainerNode,
                          FullyConnectedNode,
                          ReLUNode,
                          SGDNode)

floatX = theano.config.floatX


def test_node_constructor_arguments():

    class foo(Node, fields.Fields.a.b.c):
        pass

    assert foo.constructor_arguments() == ["a", "b", "c"]


def test_node_to_from_architecture_data():
    class foo(Node, fields.Fields.a.b.c):
        pass

    f = foo(3, 4, 5)
    assert f == f.__class__.from_architecture_data(f.to_architecture_data())
    assert f == f.from_architecture_data(f.to_architecture_data())


def test_architecture_test_node_copy():
    class foo(Node, fields.Fields.a.b.c):
        pass

    f = foo(3, 4, 5)
    assert f == f.architecture_copy()


def test_identity_network():
    input_node = InputNode("foo", (3, 4, 5))
    network = input_node.build()
    fn = network.function(["foo"], ["foo"])
    x = np.random.rand(3, 4, 5).astype(floatX)
    assert np.allclose(fn(x), x)


def test_sequential_identity_network():
    nodes = [
        InputNode("foo", (3, 4, 5)),
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
    current_node = InputNode("foo", (3, 4, 5))
    for name in map(str, range(10)):
        current_node = SequentialNode("sequential" + name,
                                      [current_node,
                                       IdentityNode("identity" + name)])
    network = current_node.build()
    fn = network.function(["foo"], ["sequential9"])
    x = np.random.rand(3, 4, 5).astype(floatX)
    assert np.allclose(fn(x), x)

# if False:
#     # NOTE: ugly
#     import pylab
#     nx.draw_networkx(
#         network.graph.computation_graph,
#         nx.spring_layout(network.graph.computation_graph),
#         node_size=500)
#     pylab.show()
# if False:
#     # plot computation_graph
#     import pylab
#     nx.draw_networkx(
#         network.graph.computation_graph,
#         nx.graphviz_layout(network.graph.computation_graph),
#         node_size=500)
#     pylab.show()
# if False:
#     # plot architectural_tree
#     import pylab
#     nx.draw_networkx(
#         network.graph.architectural_tree,
#         nx.graphviz_layout(network.graph.architectural_tree),
#         node_size=500)
#     pylab.show()


def test_toy_updater_node():

    class ToyUpdaterNode(Node, fields.Fields.name):

        """
        example node to test compute_update_deltas
        """

        def compute_output(self):
            shape = (2, 3, 4)
            self.create_variable(
                name="state",
                shape=shape,
                is_shared=True,
                tags=["state"],
            )
            init_value = np.arange(np.prod(shape)).reshape(
                *shape).astype(floatX)
            self.state.value = init_value
            return dict(
                default=self.state
            )

        def compute_update_deltas(self):
            return UpdateDeltas({
                self.state.variable: 42
            })

    network = ToyUpdaterNode("a").build()
    fn1 = network.function([], ["a"])
    init_value = fn1()
    fn2 = network.function([], ["a"], generate_updates=True)
    assert np.allclose(init_value, fn2())
    assert np.allclose(init_value[0] + 42, fn2())
    assert np.allclose(init_value[0] + 84, fn1())
    assert np.allclose(init_value[0] + 84, fn1())


def test_hyperparameter_node():
    input_node = InputNode("a", (3, 4, 5))
    hp_node = HyperparameterNode("b", input_node, foo=3, bar=2)
    network = hp_node.build()
    assert network.get_hyperparameter("foo") == 3
    a_node = network.graph.name_to_node["a"]
    assert a_node.get_hyperparameter("foo") is None
    assert a_node.find_hyperparameter("foo") == 3


class OnesInitialization(treeano.SharedInitialization):

    def initialize_value(self, var):
        return np.ones(var.shape).astype(var.dtype)


def test_ones_initialization():
    np.random.seed(42)

    class DummyNode(Node):

        def __init__(self):
            self.name = "dummy"

        def get_hyperparameter(self, hyperparameter_name):
            if hyperparameter_name == "shared_initializations":
                return [OnesInitialization()]

        def compute_output(self):
            self.create_variable(
                "foo",
                is_shared=True,
                shape=(1, 2, 3),
            )
            return dict(
                default=self.foo,
            )

    network = DummyNode().build()
    fn = network.function([], ["dummy"])
    assert np.allclose(fn(), np.ones((1, 2, 3)).astype(floatX))


def test_fully_connected_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", (3, 4, 5)),
        FullyConnectedNode("b"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=14,
                                 shared_initializations=[OnesInitialization()])
    network = hp_node.build()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
    assert np.allclose(fn(x), res)


def test_fully_connected_and_relu_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", (3, 4, 5)),
        FullyConnectedNode("b"),
        ReLUNode("e"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=14,
                                 shared_initializations=[OnesInitialization()])
    network = hp_node.build()
    fn = network.function(["a"], ["d"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
    assert np.allclose(fn(x), np.clip(res, 0, np.inf))


def test_glorot_uniform_initialization():
    np.random.seed(42)
    nodes = [
        InputNode("a", (3, 4, 5)),
        FullyConnectedNode("b"),
        ReLUNode("e"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=1000,
                                 shared_initializations=[GlorotUniform()])
    network = hp_node.build()
    fc_node = network.node.nodes[1]
    W_value = fc_node.W.value
    assert np.allclose(0, W_value.mean(), atol=1e-2)
    assert np.allclose(np.sqrt(2.0 / (20 + 1000)), W_value.std(), atol=1e-2)
    assert np.allclose(np.zeros(1000), fc_node.b.value)


def test_cost_node():
    np.random.seed(42)
    network = HyperparameterNode(
        "g",
        ContainerNode("f", [
            SequentialNode("e", [
                InputNode("input", (3, 4, 5)),
                FullyConnectedNode("b"),
                ReLUNode("c"),
                CostNode("cost", "target"),
            ]),
            InputNode("target", (3, 14)),
        ]),
        num_units=14,
        loss_function=lasagne.objectives.mse,
        shared_initializations=[OnesInitialization()]
    ).build()
    fn = network.function(["input", "target"], ["cost"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
    res = np.clip(res, 0, np.inf)
    y = np.random.randn(3, 14).astype(floatX)
    res = np.mean((y - res) ** 2)
    assert np.allclose(fn(x, y), res)


def test_update_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", (3, 4, 5)),
        FullyConnectedNode("b"),
        ReLUNode("e"),
    ]
    sequential = SequentialNode("c", nodes)
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=1000,
                                 shared_initializations=[GlorotUniform()])
    network = hp_node.build()
    fc_node = network.node.nodes[1]
    W_value = fc_node.W.value
    assert np.allclose(0, W_value.mean(), atol=1e-2)
    assert np.allclose(np.sqrt(2.0 / (20 + 1000)), W_value.std(), atol=1e-2)
    assert np.allclose(np.zeros(1000), fc_node.b.value)


def test_sgd_node():
    np.random.seed(42)
    network = HyperparameterNode(
        "g",
        SGDNode("sgd",
                ContainerNode("f", [
                    SequentialNode("e", [
                        InputNode("input", (3, 4, 5)),
                        FullyConnectedNode("b"),
                        ReLUNode("c"),
                        CostNode("cost", "target"),
                    ]),
                    InputNode("target", (3, 14)),
                ])),
        num_units=14,
        loss_function=lasagne.objectives.mse,
        shared_initializations=[OnesInitialization()],
        cost_reference="cost",
        learning_rate=0.01,
    ).build()
    fn = network.function(["input", "target"], ["cost"])
    fn2 = network.function(["input", "target"],
                           ["cost"],
                           generate_updates=True)
    x = np.random.randn(3, 4, 5).astype(floatX)
    y = np.random.randn(3, 14).astype(floatX)
    initial_cost = fn(x, y)
    next_cost = fn(x, y)
    assert np.allclose(initial_cost, next_cost)
    prev_cost = fn2(x, y)
    for _ in range(10):
        current_cost = fn2(x, y)
        assert prev_cost > current_cost
        prev_cost = current_cost
