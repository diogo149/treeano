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

    fn = network.function(["i"], ["s"])
    x = np.random.randn(3, 2, 17, 12).astype(fX)
    res = fn(x)[0]
    nt.assert_equal(network["s"].get_variable("default").shape,
                    res.shape)
