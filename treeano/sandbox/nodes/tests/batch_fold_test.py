from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import batch_fold as bf


fX = theano.config.floatX


def test_fold_axis_into_batch_node_serialization():
    tn.check_serialization(bf.FoldAxisIntoBatchNode("a"))


def test_fold_unfold_axis_into_batch_node_serialization():
    tn.check_serialization(
        bf.FoldUnfoldAxisIntoBatchNode("a", tn.IdentityNode("i")))


def test_add_axis_node_serialization():
    tn.check_serialization(bf.AddAxisNode("a"))


def test_remove_axis_node_serialization():
    tn.check_serialization(bf.RemoveAxisNode("b"))


def test_fold_unfold_axis_into_batch_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3, 4, 5)),
         bf.FoldUnfoldAxisIntoBatchNode(
             "fu1",
             tn.SequentialNode(
                 "s2",
                 [tn.IdentityNode("i1"),
                  bf.FoldUnfoldAxisIntoBatchNode(
                      "fu2",
                      tn.SequentialNode(
                          "s3",
                          [tn.IdentityNode("i2"),
                           tn.DenseNode("d", num_units=11)]),
                      axis=1)]),
             axis=3)]
    ).network()

    fn = network.function(["i"], ["i1", "i2", "fu2", "fu1"])
    x = np.zeros((2, 3, 4, 5), dtype=fX)
    i1, i2, fu2, fu1 = fn(x)
    nt.assert_equal((10, 3, 4), i1.shape)
    nt.assert_equal((30, 4), i2.shape)
    nt.assert_equal((10, 3, 11), fu2.shape)
    nt.assert_equal((2, 3, 11, 5), fu1.shape)


def test_fold_unfold_axis_into_batch_node2():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3, 4, 5)),
         bf.FoldUnfoldAxisIntoBatchNode(
             "fu",
             tn.IdentityNode("id"),
             axis=2)]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.zeros((2, 3, 4, 5), dtype=fX)
    nt.assert_equal(x.shape, fn(x)[0].shape)
    nt.assert_equal(x.shape, network["s"].get_vw("default").shape)


def test_add_remove_axis_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3, 4)),
         bf.AddAxisNode("a1", axis=3),
         bf.AddAxisNode("a2", axis=1),
         bf.RemoveAxisNode("r1", axis=1),
         bf.AddAxisNode("a3", axis=0),
         bf.RemoveAxisNode("r2", axis=4),
         bf.RemoveAxisNode("r3", axis=0)]
    ).network()

    fn = network.function(["i"], ["a1", "a2", "r1", "a3", "r2", "r3"])
    x = np.zeros((2, 3, 4), dtype=fX)
    a1, a2, r1, a3, r2, r3 = fn(x)
    nt.assert_equal((2, 3, 4, 1), a1.shape)
    nt.assert_equal((2, 1, 3, 4, 1), a2.shape)
    nt.assert_equal((2, 3, 4, 1), r1.shape)
    nt.assert_equal((1, 2, 3, 4, 1), a3.shape)
    nt.assert_equal((1, 2, 3, 4), r2.shape)
    nt.assert_equal((2, 3, 4), r3.shape)


def test_remove_axis_node():
    # testing that it works on non-broadcastable dims
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 1)),
         bf.RemoveAxisNode("r1", axis=2),
         bf.RemoveAxisNode("r2", axis=1),
         bf.RemoveAxisNode("r3", axis=0)]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.zeros((1, 1, 1), dtype=fX)
    fn(x)


def test_split_axis_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(5, 12, 7)),
         bf.SplitAxisNode("r1", axis=1, shape=(3, -1, 1, 2))]
    ).network()

    np.testing.assert_equal((5, 3, None, 1, 2, 7),
                            network["s"].get_vw("default").shape)
    fn = network.function(["i"], ["s"])
    np.testing.assert_equal((5, 3, 2, 1, 2, 7),
                            fn(np.zeros((5, 12, 7), dtype=fX))[0].shape)
