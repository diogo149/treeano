from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import interval_relu as irelu


fX = theano.config.floatX


def test_interval_relu_node_serialization():
    tn.check_serialization(irelu.IntervalReLUNode("a"))


def test_interval_relu_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 5)),
         irelu.IntervalReLUNode("a")]
    ).network()

    fn = network.function(["i"], ["s"])
    x = -1 * np.ones((1, 5), dtype=fX)
    ans = np.array([[0, -0.25, -0.5, -0.75, -1]], dtype=fX)
    np.testing.assert_allclose(ans, fn(x)[0], rtol=1e-5)
