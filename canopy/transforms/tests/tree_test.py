from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy


fX = theano.config.floatX


def test_remove_nodes():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.HyperparameterNode(
             "hp1",
             tn.HyperparameterNode(
                 "hp2",
                 tn.AddConstantNode("ac"),
                 value=1
             ),
             value=2
        )]).network()
    fn1 = network1.function(["i"], ["seq"])
    nt.assert_equal(1, fn1(0)[0])
    network2 = canopy.transforms.remove_nodes(network1,
                                              {"hp2"},
                                              keep_child=True)
    fn2 = network2.function(["i"], ["seq"])
    nt.assert_equal(2, fn2(0)[0])
    network3 = canopy.transforms.remove_nodes(network1, {"ac"})
    fn3 = network3.function(["i"], ["seq"])
    nt.assert_equal(0, fn3(0)[0])


def test_remove_subtree():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.HyperparameterNode(
             "hp1",
             tn.HyperparameterNode(
                 "hp2",
                 tn.AddConstantNode("ac"),
                 value=1
             ),
             value=2
        )]).network()
    fn1 = network1.function(["i"], ["seq"])
    nt.assert_equal(1, fn1(0)[0])
    network2 = canopy.transforms.remove_subtree(network1, {"hp2"})
    fn2 = network2.function(["i"], ["seq"])
    nt.assert_equal(0, fn2(0)[0])
    network3 = canopy.transforms.remove_subtree(network1, {"hp1"})
    fn3 = network3.function(["i"], ["seq"])
    nt.assert_equal(0, fn3(0)[0])
    network4 = canopy.transforms.remove_subtree(network1, {"ac"})
    fn4 = network4.function(["i"], ["seq"])
    nt.assert_equal(0, fn4(0)[0])


def test_remove_parent():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.HyperparameterNode(
             "hp1",
             tn.HyperparameterNode(
                 "hp2",
                 tn.AddConstantNode("ac"),
                 value=1
             ),
             value=2
        )]).network()
    fn1 = network1.function(["i"], ["seq"])
    nt.assert_equal(1, fn1(0)[0])
    network2 = canopy.transforms.remove_parent(network1, {"ac"})
    fn2 = network2.function(["i"], ["seq"])
    nt.assert_equal(2, fn2(0)[0])

    network3 = canopy.transforms.remove_parent(network1, {"i"})

    @nt.raises(Exception)
    def fails(name):
        network3.function(["i"], [name])

    # testing that these nodes are removed
    fails("ac")
    fails("seq")
    network3.function(["i"], ["i"])


def test_add_hyperparameters():
    network1 = treeano.Network(
        tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=()),
             tn.AddConstantNode("ac")]),
        default_hyperparameters={"value": 2})
    fn1 = network1.function(["i"], ["ac"])
    nt.assert_equal(2, fn1(0)[0])
    network2 = canopy.transforms.add_hyperparameters(
        network1, "hp", dict(value=3))
    print(network2.root_node)
    fn2a = network2.function(["i"], ["ac"])
    nt.assert_equal(3, fn2a(0)[0])
    fn2b = network2.function(["i"], ["hp"])
    nt.assert_equal(3, fn2b(0)[0])


def test_remove_parents():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.HyperparameterNode(
             "hp1",
             tn.HyperparameterNode(
                 "hp2",
                 tn.AddConstantNode("ac"),
                 value=1
             ),
             value=2
        )]).network()

    network2 = canopy.transforms.remove_parents(network1, "ac")

    nt.assert_equal(tn.AddConstantNode("ac"), network2.root_node)


def test_move_node():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.HyperparameterNode(
             "hp1",
             tn.HyperparameterNode(
                 "hp2",
                 tn.AddConstantNode("ac"),
                 value=1
             ),
             value=2
        )]).network()

    network2 = canopy.transforms.move_node(network1, "ac", "hp1")

    ans = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.AddConstantNode("ac")])

    nt.assert_equal(ans, network2.root_node)
