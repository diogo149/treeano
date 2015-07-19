import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import wta_sparisty as wta


fX = theano.config.floatX


def test_wta_spatial_sparsity_node_serialization():
    tn.check_serialization(wta.WTASpatialSparsityNode("a"))


def test_wta_sparsity_node_serialization():
    tn.check_serialization(wta.WTASparsityNode("a"))


def test_wta_spatial_sparsity_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 2, 2, 2)),
         wta.WTASpatialSparsityNode("a")]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(16).reshape(2, 2, 2, 2).astype(fX)
    ans = x.copy()
    ans[..., 0] = 0
    ans[..., 0, :] = 0
    np.testing.assert_allclose(fn(x)[0],
                               ans)


def test_wta_sparsity_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 2, 2, 2)),
         wta.WTASparsityNode("a", percentile=0.5)]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.arange(16).reshape(2, 2, 2, 2).astype(fX)
    ans = x.copy()
    ans[..., 0] = 0
    ans[..., 0, :] = 0
    ans[0] = 0
    res = fn(x)[0]
    np.testing.assert_allclose(res, ans)
