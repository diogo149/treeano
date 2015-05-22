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
