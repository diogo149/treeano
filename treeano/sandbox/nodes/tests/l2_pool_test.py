import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import l2_pool

fX = theano.config.floatX


def test_l2_pool_2d_node_serialization():
    # NOTE: serialization converts pool_size to list, so must be a list
    tn.check_serialization(l2_pool.L2Pool2DNode("a"))


def test_l2_pool_2d_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 4, 4)),
         l2_pool.L2Pool2DNode("p", pool_size=(2, 2))]).network()
    fn = network.function(["i"], ["s"])
    x = np.array([[[[3, 4, 1, 2],
                    [0, 0, 3, 4],
                    [1, 1, -1, 1],
                    [1, 1, 1, -1]]]], dtype=fX)
    ans = np.array([[[[5, np.linalg.norm([1, 2, 3, 4])],
                      [2, 2]]]], dtype=fX)
    np.testing.assert_allclose(ans, fn(x)[0])
