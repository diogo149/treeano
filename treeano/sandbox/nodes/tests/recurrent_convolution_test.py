import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import recurrent_convolution as rcl

fX = theano.config.floatX


def test_default_recurrent_conv_2d_node_serialization():
    tn.check_serialization(rcl.DefaultRecurrentConv2DNode("a"))


if "gpu" in theano.config.device:

    def test_default_recurrent_conv_2d_node():
        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(3, 4, 5, 6)),
             rcl.DefaultRecurrentConv2DNode("a",
                                            num_filters=7,
                                            filter_size=(3, 3),
                                            pad="same")]
        ).network()
        fn = network.function(["i"], ["s"])
        res = fn(np.random.randn(3, 4, 5, 6).astype(fX))[0]
        np.testing.assert_equal((3, 7, 5, 6), res.shape)
