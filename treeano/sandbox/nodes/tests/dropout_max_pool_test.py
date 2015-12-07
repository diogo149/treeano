import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import dropout_max_pool as dmp

fX = theano.config.floatX


def test_dropout_max_pool_2d_node_serialization():
    tn.check_serialization(dmp.DropoutMaxPool2DNode("a"))


def test_dropout_max_pool_2d_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 2, 2)),
         dmp.DropoutMaxPool2DNode("a",
                                  pool_size=(2, 2),
                                  dropout_probability=0.3,
                                  deterministic=True)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.arange(4).astype(fX).reshape(1, 1, 2, 2)
    ans = np.array([[[[1 * 0.7 * 0.3 ** 2 + 2 * 0.7 * 0.3 + 3 * 0.7]]]],
                   dtype=fX)
    np.testing.assert_allclose(fn(x)[0],
                               ans,
                               rtol=1e-5)
    nt.assert_equal(network["s"].get_vw("default").shape,
                    ans.shape)


def test_dropout_max_pool_2d_node2():
    # testing that stochastic version works
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         dmp.DropoutMaxPool2DNode("a", pool_size=(2, 2))]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    fn(x)
