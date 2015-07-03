import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_relu_node_serialization():
    tn.check_serialization(tn.ReLUNode("a"))


def test_softmax_node_serialization():
    tn.check_serialization(tn.SoftmaxNode("a"))


def test_resqrt_node_serialization():
    tn.check_serialization(tn.ReSQRTNode("a"))


def test_channel_out_node_serialization():
    tn.check_serialization(tn.ChannelOutNode("a"))


def test_channel_out_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 15)),
         tn.ChannelOutNode("m", num_pieces=5)]).network()

    fn = network.function(["i"], ["m"])
    x = np.arange(15).astype(fX).reshape(1, 15)
    ans = np.zeros_like(x)
    ans[:, [4, 9, 14]] = [4, 9, 14]
    np.testing.assert_equal(fn(x)[0],
                            ans)
