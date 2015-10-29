from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import input_scaling


fX = theano.config.floatX


def test_clip_scaling_node_serialization():
    tn.check_serialization(input_scaling.ClipScalingNode("a"))


def test_clip_scaling_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(None, 2)),
         input_scaling.ClipScalingNode("c",
                                       mins=np.array([0, 1]),
                                       maxs=np.array([2, 3]))]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.arange(6).reshape(3, 2).astype(fX)
    res = fn(x)[0]
    ans = np.array([[0, 0, 0.5, 0],
                    [1, 0.5, 1, 1],
                    [1, 1, 1, 1]],)
    np.testing.assert_allclose(ans, res)


def test_clip_scaling_node_learnable():
    # just testing that it runs, not that it learns
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(None, 2)),
         input_scaling.ClipScalingNode("c",
                                       mins=np.array([0, 1]),
                                       maxs=np.array([2, 3]),
                                       learnable=True)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.arange(6).reshape(3, 2).astype(fX)
    res = fn(x)[0]
    ans = np.array([[0, 0, 0.5, 0],
                    [1, 0.5, 1, 1],
                    [1, 1, 1, 1]],)
    np.testing.assert_allclose(ans, res)
