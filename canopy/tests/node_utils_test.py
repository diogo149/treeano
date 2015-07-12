import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy

fX = theano.config.floatX


def test_postwalk_node():
    names = []

    def f(node):
        names.append(node.name)
        return node

    node = tn.HyperparameterNode(
        "1",
        tn.HyperparameterNode(
            "2",
            tn.IdentityNode("3")))
    canopy.node_utils.postwalk_node(node, f)
    nt.assert_equal(names, ["3", "2", "1"])


def test_suffix_node():
    node1 = tn.HyperparameterNode(
        "1",
        tn.HyperparameterNode(
            "2",
            tn.IdentityNode("3")))
    node2 = tn.HyperparameterNode(
        "1_foo",
        tn.HyperparameterNode(
            "2_foo",
            tn.IdentityNode("3_foo")))
    nt.assert_equal(canopy.node_utils.suffix_node(node1, "_foo"),
                    node2)
