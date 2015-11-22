import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import stochastic_pooling

fX = theano.config.floatX


def test_stochastic_pool_2d_node_serialization():
    tn.check_serialization(stochastic_pooling.StochasticPool2DNode("a"))


def test_stochastic_pool_2d_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         stochastic_pooling.StochasticPool2DNode("m",
                                                 pool_size=(2, 2),
                                                 deterministic=True)]
    ).network()
    fn = network.function(["i"], ["m"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    pre_pool = np.array([[[[[0, 1, 4, 5], [2, 3, 6, 7]],
                           [[8, 9, 12, 13], [10, 11, 14, 15]]]]], dtype=fX)
    ans = ((pre_pool ** 2) / pre_pool.sum(axis=-1)[..., None]).sum(axis=-1)
    np.testing.assert_allclose(fn(x)[0],
                               ans,
                               rtol=1e-5)
    nt.assert_equal(network["m"].get_vw("default").shape,
                    ans.shape)


def test_stochastic_pool_2d_node2():
    # testing that stochastic version works
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         stochastic_pooling.StochasticPool2DNode("m",
                                                 pool_size=(2, 2))]
    ).network()
    fn = network.function(["i"], ["m"])
    x = np.arange(16).astype(fX).reshape(1, 1, 4, 4)
    fn(x)
