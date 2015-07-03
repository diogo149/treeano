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
