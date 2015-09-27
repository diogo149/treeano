import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_repeat_n_d_node_serialization():
    tn.check_serialization(tn.RepeatNDNode("a"))


def test_repeat_n_d_node_serialization():
    tn.check_serialization(tn.SparseUpsampleNode("a"))


def test_repeat_n_d_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(3,)),
         tn.RepeatNDNode("r", upsample_factor=(2,))]).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(3).astype(fX)
    np.testing.assert_equal(np.array([0, 0, 1, 1, 2, 2], dtype=fX),
                            fn(x)[0])


def test_repeat_n_d_node2():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.RepeatNDNode("r", upsample_factor=(1, 1, 1))]).network()

    fn = network.function(["i"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_equal(x,
                            fn(x)[0])


def test_repeat_n_d_node3():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3)),
         tn.RepeatNDNode("r", upsample_factor=(2, 1))]).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(6).astype(fX).reshape(2, 3)
    np.testing.assert_equal(np.array([[0, 1, 2],
                                      [0, 1, 2],
                                      [3, 4, 5],
                                      [3, 4, 5]]),
                            fn(x)[0])


def test_repeat_n_d_node4():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3)),
         tn.RepeatNDNode("r", upsample_factor=(1, 2))]).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(6).astype(fX).reshape(2, 3)
    np.testing.assert_equal(np.array([[0, 0, 1, 1, 2, 2],
                                      [3, 3, 4, 4, 5, 5]]),
                            fn(x)[0])


def test_sparse_upsample_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3)),
         tn.SparseUpsampleNode("r", upsample_factor=(1, 2))]).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(6).astype(fX).reshape(2, 3)
    np.testing.assert_equal(np.array([[0, 0, 1, 0, 2, 0],
                                      [3, 0, 4, 0, 5, 0]]),
                            fn(x)[0])
