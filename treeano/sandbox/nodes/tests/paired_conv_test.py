import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import paired_conv

fX = theano.config.floatX


def test_paired_conv_2d_with_bias_node_serialization():
    tn.check_serialization(paired_conv.PairedConvNode("a", {}))


def test_paired_conv_2d_with_bias_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(3, 4, 5, 6)),
         paired_conv.PairedConvNode(
             "c",
             {"conv": tn.Conv2DWithBiasNode("c_conv"),
              "separator": tn.IdentityNode("sep")},
             filter_size=(2, 2),
             num_filters=7,
             pad="same")]
    ).network()
    fn = network.function(["i"], ["s"])
    res = fn(np.random.randn(3, 4, 5, 6).astype(fX))[0]
    np.testing.assert_equal((3, 7, 5, 6), res.shape)
