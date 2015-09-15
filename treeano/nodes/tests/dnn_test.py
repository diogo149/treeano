import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_dnn_pool_node_serialization():
    tn.check_serialization(tn.DnnPoolNode("a"))
