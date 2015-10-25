import nose.tools as nt
import numpy as np
import theano

import treeano.nodes as tn

fX = theano.config.floatX


def test_sequential_node_serialization():
    tn.check_serialization(tn.SequentialNode("a", []))
    tn.check_serialization(tn.SequentialNode(
        "a",
        [tn.SequentialNode("b", []),
         tn.SequentialNode("c", [])]))


def test_auxiliary_node_serialization():
    tn.check_serialization(tn.AuxiliaryNode("a", tn.IdentityNode("b")))


def test_graph_node_serialization():
    tn.check_serialization(tn.GraphNode("a", [[tn.IdentityNode("b")],
                                              [{"from": "b", "to": "a"}]]))


@nt.raises(AssertionError)
def test_container_node_raises():
    network = tn.SequentialNode(
        "s",
        [tn.ContainerNode("c", []),
         tn.IdentityNode("i")
         ]).network()
    fn = network.function([], ["i"])
    fn()


def test_auxiliary_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.AuxiliaryNode("a", tn.MultiplyConstantNode("m", value=2))]
    ).network()
    fn = network.function(["i"], ["s", "a", "m"])
    np.testing.assert_equal(np.array(fn(3.2)),
                            np.array([3.2, 3.2, 6.4], dtype=fX))


def test_graph_node():
    network = tn.GraphNode(
        "g1",
        [[tn.InputNode("i", shape=()),
          tn.GraphNode("g2",
                       [(tn.MultiplyConstantNode("m1", value=2),
                         tn.AddConstantNode("a1", value=2)),
                        [{"to": "a1"},
                         {"from": "a1", "to": "m1"},
                         {"from": "m1", "to_key": "foo"}]],
                       output_key="foo")],
         [{"from": "i", "to": "g2"},
          {"from": "g2", "to_key": "bar"}]],
        output_key="bar"
    ).network()
    fn = network.function(["i"], ["a1", "m1", "g1", "g2"])
    nt.assert_equal([5, 10, 10, 10], fn(3))


def test_graph_node_no_output_key():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.GraphNode("g",
                      [(tn.MultiplyConstantNode("m1", value=2),
                        tn.AddConstantNode("a1", value=2)),
                       [{"to": "a1"},
                        {"from": "a1", "to": "m1"},
                        {"from": "m1",  "to_key": "foo"}]])]
    ).network()
    fn = network.function(["i"], ["s"])
    nt.assert_equal([3], fn(3))


def test_graph_node_no_input():
    network = tn.GraphNode(
        "g",
        [(tn.InputNode("i", shape=()),
          tn.MultiplyConstantNode("m1", value=2),
          tn.AddConstantNode("a1", value=2)),
         [{"from": "i", "to": "a1"},
          {"from": "a1", "to": "m1"},
          {"from": "m1"}]]
    ).network()
    fn = network.function(["i"], ["g"])
    nt.assert_equal([10], fn(3))
