import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import spp_net

fX = theano.config.floatX


def test_spatial_pyramid_pooling_node_serialization():
    tn.check_serialization(spp_net.SpatialPyramidPoolingNode("a"))


def test_spatial_pyramid_pooling_node():
    # only testing size
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(3, 2, 17, 12)),
         spp_net.SpatialPyramidPoolingNode("spp", spp_levels=[(1, 1),
                                                              (2, 2),
                                                              (3, 4),
                                                              (5, 5),
                                                              (17, 12)])]
    ).network()

    ans_shape = (3, 2 * (1 * 1 + 2 * 2 + 3 * 4 + 5 * 5 + 17 * 12))
    fn = network.function(["i"], ["s"])
    x = np.random.randn(3, 2, 17, 12).astype(fX)
    res = fn(x)[0]
    nt.assert_equal(network["s"].get_vw("default").shape,
                    ans_shape)
    nt.assert_equal(res.shape,
                    ans_shape)


# this currently doesn't work because
@nt.raises(AssertionError)
def test_spatial_pyramid_pooling_node_symbolic():
    # only testing size
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(None, 2, None, None)),
         spp_net.SpatialPyramidPoolingNode("spp", spp_levels=[(1, 1),
                                                              (2, 2),
                                                              (3, 4),
                                                              (5, 5),
                                                              (17, 12)])]
    ).network()

    fn = network.function(["i"], ["s"])
    ans_shape = (3, 2 * (1 * 1 + 2 * 2 + 3 * 4 + 5 * 5 + 17 * 12))
    x1 = np.random.randn(3, 2, 17, 12).astype(fX)
    nt.assert_equal(ans_shape,
                    fn(x1)[0].shape)
    x2 = np.random.randn(100, 2, 177, 123).astype(fX)
    nt.assert_equal(ans_shape,
                    fn(x2)[0].shape)
