import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import mixed_pooling

fX = theano.config.floatX


def test_mixed_pool_node_serialization():
    tn.check_serialization(mixed_pooling.MixedPoolNode("a"))


def test_gated_pool_2d_node_serialization():
    tn.check_serialization(mixed_pooling.GatedPool2DNode("a"))
