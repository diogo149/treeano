import numpy as np
import theano

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
