import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import spatial_attention

fX = theano.config.floatX


def test_spatial_feature_point_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 2, 2, 3)),
         spatial_attention.SpatialFeaturePointNode("fp")]
    ).network()

    fn = network.function(["i"], ["s"])

    x = np.zeros((2, 2, 2, 3), dtype=fX)
    idxs = np.array([[[0, 0],
                      [1, 0]],
                     [[0, 1],
                      [1, 2]]],
                    dtype=fX)
    ans = idxs / np.array([1, 2], dtype=fX)[None, None]
    for batch in range(2):
        for channel in range(2):
            i, j = idxs[batch, channel]
            x[batch, channel, i, j] = 1

    np.testing.assert_allclose(ans,
                               fn(x)[0],
                               rtol=1e-5,
                               atol=1e-8)


def test_pairwise_distance_node():
    # NOTE: only tests shape calculation
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 2, 2, 3)),
         spatial_attention.SpatialFeaturePointNode("fp"),
         spatial_attention.PairwiseDistanceNode("pd")]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.zeros((2, 2, 2, 3), dtype=fX)

    nt.assert_equal((2, 4),
                    fn(x)[0].shape)
