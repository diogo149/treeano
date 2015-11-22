import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import deconv_upsample

fX = theano.config.floatX


def test_deconv_upsample_2d_node_serialization():
    tn.check_serialization(deconv_upsample.DeconvUpsample2DNode("a"))


if "gpu" in theano.config.device:

    def test_default_recurrent_conv_2d_node():
        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(3, 4, 5, 6)),
             deconv_upsample.DeconvUpsample2DNode(
                "a",
                num_filters=7,
                upsample_factor=(2, 2),
                filter_size=(3, 3),
            )]
        ).network()
        fn = network.function(["i"], ["s"])
        res = fn(np.random.randn(3, 4, 5, 6).astype(fX))[0]
        np.testing.assert_equal((3, 7, 10, 12), res.shape)
        np.testing.assert_equal((3, 7, 10, 12),
                                network['a'].get_vw('default').shape)
