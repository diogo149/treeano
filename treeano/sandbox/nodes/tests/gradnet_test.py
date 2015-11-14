from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import gradnet

fX = theano.config.floatX


def test_grad_net_interpolation_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 10)),
         gradnet.GradNetInterpolationNode(
             "gradnet",
             {"early": tn.ReLUNode("r"),
              "late": tn.TanhNode("t")},
             late_gate=0.5)]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.random.randn(1, 10).astype(fX)
    ans = 0.5 * np.clip(x, 0, np.inf) + 0.5 * np.tanh(x)
    np.testing.assert_allclose(ans, fn(x)[0])
