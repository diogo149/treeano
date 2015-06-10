import nose.tools as nt
import numpy as np
import theano

import treeano.nodes as tn

floatX = theano.config.floatX


def test_sequential_node_serialization():
    tn.check_serialization(tn.SequentialNode("a", []))
    tn.check_serialization(tn.SequentialNode(
        "a",
        [tn.SequentialNode("b", []),
         tn.SequentialNode("c", [])]))


@nt.raises(AssertionError)
def test_container_node_raises():
    network = tn.SequentialNode(
        "s",
        [tn.ContainerNode("c", []),
         tn.IdentityNode("i")
         ]).build()
    fn = network.function([], ["i"])
    fn()
