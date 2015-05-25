import copy

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.core
from treeano import nodes

floatX = theano.config.floatX


def test_reference_node_serialization():
    nodes.check_serialization(nodes.ReferenceNode("a"))
    nodes.check_serialization(nodes.ReferenceNode("a", reference="bar"))


def test_send_to_node_serialization():
    nodes.check_serialization(nodes.SendToNode("a"))
    nodes.check_serialization(nodes.SendToNode("a", reference="bar"))


def test_function_combine_node_serialization():
    nodes.check_serialization(nodes.FunctionCombineNode("a"))


def test_add_bias_node_serialization():
    nodes.check_serialization(nodes.AddBiasNode("a"))
    nodes.check_serialization(nodes.AddBiasNode(
        "a",
        inits=[],
        # need to make broadcastable a list because json (de)serialization
        # converts tuples to lists
        broadcastable=[True, False, True]))


def test_linear_mapping_node_serialization():
    nodes.check_serialization(nodes.LinearMappingNode("a"))
    nodes.check_serialization(nodes.LinearMappingNode("a", output_dim=3))


def test_apply_node_serialization():
    nodes.check_serialization(nodes.ApplyNode("a"))


def test_reference_node():
    network = nodes.SequentialNode("s", [
        nodes.InputNode("input1", shape=(3, 4, 5)),
        nodes.InputNode("input2", shape=(5, 4, 3)),
        nodes.ReferenceNode("ref", reference="input1"),
    ]).build()

    fn = network.function(["input1"], ["ref"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    np.testing.assert_allclose(fn(x)[0], x)


def test_send_to_node():
    network = nodes.ContainerNode("c", [
        nodes.SequentialNode(
            "s1",
            [nodes.InputNode("in", shape=(3, 4, 5)),
             nodes.SendToNode("stn1", reference="s2")]),
        nodes.SequentialNode(
            "s2",
            [nodes.SendToNode("stn2", reference="stn3")]),
        nodes.SequentialNode(
            "s3",
            [nodes.SendToNode("stn3", reference="i")]),
        nodes.IdentityNode("i"),
    ]).build()

    fn = network.function(["in"], ["i"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    np.testing.assert_allclose(fn(x)[0], x)


def test_function_combine_node():
    def fcn_network(combine_fn):
        network = nodes.ContainerNode("c", [
            nodes.SequentialNode(
                "s1",
                [nodes.InputNode("in1", shape=(3, 4, 5)),
                 nodes.SendToNode("stn1", reference="fcn", to_key="b")]),
            nodes.SequentialNode(
                "s2",
                [nodes.InputNode("in2", shape=(3, 4, 5)),
                 nodes.SendToNode("stn2", reference="fcn", to_key="a")]),
            nodes.FunctionCombineNode("fcn", combine_fn=combine_fn)
        ]).build()
        return network.function(["in1", "in2"], ["fcn"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    y = np.random.randn(3, 4, 5).astype(floatX)
    fn1 = fcn_network(lambda *args: sum(args))
    np.testing.assert_allclose(fn1(x, y)[0], x + y)
    fn2 = fcn_network(lambda x, y: x * y)
    np.testing.assert_allclose(fn2(x, y)[0], x * y)
    # testing alphabetical ordering of to_key
    # ---
    # adding other key times 0 to avoid unused input error
    fn3 = fcn_network(lambda x, y: x + 0 * y)
    np.testing.assert_allclose(fn3(x, y)[0], y)
    fn4 = fcn_network(lambda x, y: y + 0 * x)
    np.testing.assert_allclose(fn4(x, y)[0], x)


def test_network_doesnt_mutate():
    root_node = nodes.ContainerNode("c", [
        nodes.SequentialNode(
            "s1",
            [nodes.InputNode("in1", shape=(3, 4, 5)),
             nodes.SendToNode("stn1", reference="fcn", to_key="b")]),
        nodes.SequentialNode(
            "s2",
            [nodes.InputNode("in2", shape=(3, 4, 5)),
             nodes.SendToNode("stn2", reference="fcn", to_key="a")]),
        nodes.FunctionCombineNode("fcn", combine_fn=lambda *args: sum(args))
    ])
    original_dict = copy.deepcopy(root_node.__dict__)
    root_node.build()
    nt.assert_equal(original_dict,
                    root_node.__dict__)


def test_node_with_generated_children_can_serialize():
    root_node = nodes.ContainerNode("c", [
        nodes.SequentialNode(
            "s1",
            [nodes.InputNode("in1", shape=(3, 4, 5)),
             nodes.SendToNode("stn1", reference="fcn", to_key="b")]),
        nodes.SequentialNode(
            "s2",
            [nodes.InputNode("in2", shape=(3, 4, 5)),
             nodes.SendToNode("stn2", reference="fcn", to_key="a")]),
        nodes.FunctionCombineNode("fcn", combine_fn=lambda *args: sum(args))
    ])
    root_node.build()
    root2 = treeano.core.node_from_data(treeano.core.node_to_data(root_node))
    nt.assert_equal(root_node, root2)


def test_add_bias_node_broadcastable():
    def get_bias_shape(broadcastable):
        return nodes.SequentialNode("s", [
            nodes.InputNode("in", shape=(3, 4, 5)),
            (nodes.AddBiasNode("b", broadcastable=broadcastable)
             if broadcastable is not None
             else nodes.AddBiasNode("b"))
        ]).build()["b"].get_variable("bias").shape

    nt.assert_equal((1, 4, 5),
                    get_bias_shape(None))
    nt.assert_equal((1, 4, 1),
                    get_bias_shape((True, False, True)))
    nt.assert_equal((3, 1, 5),
                    get_bias_shape((False, True, False)))


@nt.raises(AssertionError)
def test_add_bias_node_broadcastable_incorrect_size1():
    nodes.SequentialNode("s", [
        nodes.InputNode("in", shape=(3, 4, 5)),
        nodes.AddBiasNode("b", broadcastable=(True, False))
    ]).build()


@nt.raises(AssertionError)
def test_add_bias_node_broadcastable_incorrect_size2():
    nodes.SequentialNode("s", [
        nodes.InputNode("in", shape=(3, 4, 5)),
        nodes.AddBiasNode("b", broadcastable=(True, False, True, False))
    ]).build()


def test_add_bias_node():
    network = nodes.SequentialNode("s", [
        nodes.InputNode("in", shape=(3, 4, 5)),
        nodes.AddBiasNode("b", broadcastable_axes=())
    ]).build()
    bias_var = network["b"].get_variable("bias")
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    y = np.random.randn(3, 4, 5).astype(floatX)
    # test that bias is 0 initially
    np.testing.assert_allclose(fn(x)[0], x)
    # set bias_var value to new value
    bias_var.value = y
    # test that adding works
    np.testing.assert_allclose(fn(x)[0], x + y)


def test_linear_mapping_node_shape():
    def get_shapes(output_dim):
        network = nodes.SequentialNode("s", [
            nodes.InputNode("in", shape=(3, 4, 5)),
            nodes.LinearMappingNode("linear", output_dim=output_dim),
        ]).build()
        weight_shape = network["linear"].get_variable("weight").shape
        output_shape = network["s"].get_variable("default").shape
        return weight_shape, output_shape

    nt.assert_equal(((5, 10), (3, 4, 10)), get_shapes(10))
    nt.assert_equal(((5, 1), (3, 4, 1)),
                    get_shapes(1))


def test_linear_mapping_node():
    network = nodes.SequentialNode("s", [
        nodes.InputNode("in", shape=(3, 4, 5)),
        nodes.LinearMappingNode("linear", output_dim=6),
    ]).build()
    weight_var = network["linear"].get_variable("weight")
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    W = np.random.randn(5, 6).astype(floatX)
    # test that weight is 0 initially
    np.testing.assert_allclose(fn(x)[0], np.zeros((3, 4, 6)))
    # set weight_var value to new value
    weight_var.value = W
    # test that adding works
    np.testing.assert_allclose(fn(x)[0], np.dot(x, W))


def test_apply_node():
    network = nodes.SequentialNode("s", [
        nodes.InputNode("in", shape=(3, 4, 5)),
        nodes.ApplyNode("a", fn=T.sum, shape_fn=lambda x: ()),
    ]).build()
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(floatX)
    np.testing.assert_allclose(fn(x)[0],
                               x.sum(),
                               rtol=1e-5)
