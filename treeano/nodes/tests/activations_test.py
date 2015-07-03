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
