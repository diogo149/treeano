import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_print_node_serialization():
    tn.check_serialization(tn.PrintNode("a"))
