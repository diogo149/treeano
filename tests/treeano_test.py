from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import lasagne

import treeano
import treeano.lasagne
from treeano import UpdateDeltas
from treeano.lasagne.initialization import GlorotUniform
from treeano.nodes import (InputNode,
                           SequentialNode,
                           IdentityNode,
                           HyperparameterNode,
                           CostNode,
                           UpdateScaleNode,
                           ContainerNode,)
from treeano.lasagne.nodes import (DenseNode,
                                   ReLUNode,
                                   SGDNode)

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


class OnesInitialization(treeano.SharedInitialization):

    def initialize_value(self, var):
        return np.ones(var.shape).astype(var.dtype)


def test_ones_initialization():
    class DummyNode(treeano.NodeImpl):

        input_keys = ()

        def get_hyperparameter(self, network, hyperparameter_name):
            if hyperparameter_name == "shared_initializations":
                return [OnesInitialization()]
            else:
                return super(DummyNode, self).get_hyperparameter(
                    network,
                    hyperparameter_name)

        def compute_output(self, network):
            network.create_variable(
                "default",
                is_shared=True,
                shape=(1, 2, 3),
            )

    network = DummyNode("dummy").build()
    fn = network.function([], ["dummy"])
    np.testing.assert_allclose(fn()[0],
                               np.ones((1, 2, 3)).astype(floatX),
                               rtol=1e-5,
                               atol=1e-8)


def test_dense_node():
    np.random.seed(42)
    nodes = [
        InputNode("a", shape=(3, 4, 5)),
        DenseNode("b"),
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
    hp_node = HyperparameterNode("d",
                                 sequential,
                                 num_units=14,
                                 shared_initializations=[OnesInitialization()])
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
                                 shared_initializations=[GlorotUniform()])
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


def test_cost_node():
    np.random.seed(42)
    network = HyperparameterNode(
        "g",
        ContainerNode("f", [
            SequentialNode("e", [
                InputNode("input", shape=(3, 4, 5)),
                DenseNode("b"),
                ReLUNode("c"),
                CostNode("cost", reference="target"),
            ]),
            InputNode("target", shape=(3, 14)),
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
    np.testing.assert_allclose(fn(x, y)[0],
                               res,
                               rtol=1e-5,
                               atol=1e-8)


def test_update_node():
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
                                 shared_initializations=[GlorotUniform()])
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


def test_sgd_node():
    np.random.seed(42)
    network = HyperparameterNode(
        "g",
        SGDNode("sgd",
                ContainerNode("f", [
                    SequentialNode("e", [
                        InputNode("input", shape=(3, 4, 5)),
                        DenseNode("b"),
                        ReLUNode("c"),
                        CostNode("cost", reference="target"),
                    ]),
                    InputNode("target", shape=(3, 14)),
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
        assert prev_cost > current_cost
        prev_cost = current_cost


def test_update_scale_node():

    class ConstantUpdaterNode(treeano.Wrapper1NodeImpl):

        hyperparameter_names = ("value",)

        def mutate_update_deltas(self, network, update_deltas):
            value = network.find_hyperparameter(["value"])
            parameters = network.find_variables_in_subtree(["parameter"])
            for parameter in parameters:
                update_deltas[parameter.variable] = value

    # testing constant updater
    network = ConstantUpdaterNode(
        "cun",
        SequentialNode("seq", [
            InputNode("i", shape=(1, 2, 3)),
            DenseNode("fc", num_units=5)
        ]),
        value=5,
    ).build()
    ud = network.update_deltas
    assert ud[network["fc"].get_variable("W").variable] == 5

    network = ConstantUpdaterNode(
        "cun",
        SequentialNode("seq", [
            InputNode("i", shape=(1, 2, 3)),
            UpdateScaleNode("usn",
                            DenseNode("fc", num_units=5),
                            scale_factor=-2)
        ]),
        value=5,
    ).build()
    ud = network.update_deltas
    assert ud[network["fc"].get_variable("W").variable] == -10
