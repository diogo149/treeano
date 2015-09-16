import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_conv_parse_pad():
    tests = [
        [(3, 4, 5), "full", (2, 3, 4)],
        [(3, 4, 5), "valid", (0, 0, 0)],
        [(3, 5, 7), "same", (1, 2, 3)],
        [(1, 1), "same", (0, 0)],
        [(1, 1), (3, 3), (3, 3)],
    ]
    for filter_size, pad, ans in tests:
        nt.assert_equal(ans, tn.conv.conv_parse_pad(filter_size, pad))

    fails_fn = nt.raises(AssertionError)(tn.conv.conv_parse_pad)
    fails_fn((2,), "same")
    fails_fn((2, 3), (1, 2, 3))


def test_conv_2d_node_serialization():
    tn.check_serialization(tn.Conv2DNode("a"))
