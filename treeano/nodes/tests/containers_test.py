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
         tn.AuxiliaryNode("a", tn.toy.MultiplyConstantNode("m", value=2))]
    ).network()
    fn = network.function(["i"], ["s", "a", "m"])
    np.testing.assert_equal(np.array(fn(3.2)),
                            np.array([3.2, 3.2, 6.4], dtype=fX))
