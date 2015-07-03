import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_feature_pool_node_serialization():
    tn.check_serialization(tn.FeaturePoolNode("a"))


def test_maxout_node_serialization():
    tn.check_serialization(tn.MaxoutNode("a"))


def test_pool_2d_node_serialization():
    tn.check_serialization(tn.Pool2DNode("a"))


def test_mean_pool_2d_node_serialization():
    tn.check_serialization(tn.MeanPool2DNode("a"))


def test_maxout_hyperparameters():
    nt.assert_equal(
        set(tn.FeaturePoolNode.hyperparameter_names),
        set(tn.MaxoutNode.hyperparameter_names + ("pool_function",)))


def test_maxout_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 15)),
         tn.MaxoutNode("m", num_pieces=5)]).network()

    fn = network.function(["i"], ["m"])
    x = np.arange(15).astype(fX).reshape(1, 15)
    np.testing.assert_equal(fn(x)[0],
                            np.array([[4, 9, 14]], dtype=fX))


def test_mean_pool_2d_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         tn.MeanPool2DNode("m", pool_size=(2, 2))]).network()
    fn = network.function(["i"], ["m"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    ans = np.array([[[[0 + 1 + 4 + 5, 2 + 3 + 6 + 7],
                      [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]]]], dtype=fX) / 4
    np.testing.assert_equal(fn(x)[0], ans)
    nt.assert_equal(network["m"].get_variable("default").shape,
                    ans.shape)
