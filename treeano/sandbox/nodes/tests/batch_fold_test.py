from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import batch_fold as bf


fX = theano.config.floatX


def test_fold_axis_into_batch_node_serialization():
    tn.check_serialization(bf.FoldAxisIntoBatchNode("a"))


def test_fold_unfold_axis_into_batch_node_serialization():
    tn.check_serialization(
        bf.FoldUnfoldAxisIntoBatchNode("a", tn.IdentityNode("i")))


def test_fold_unfold_axis_into_batch_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(2, 3, 4, 5)),
         bf.FoldUnfoldAxisIntoBatchNode(
             "fu1",
             tn.SequentialNode(
                 "s2",
                 [tn.IdentityNode("i1"),
                  bf.FoldUnfoldAxisIntoBatchNode(
                      "fu2",
                      tn.SequentialNode(
                          "s3",
                          [tn.IdentityNode("i2"),
                           tn.DenseNode("d", num_units=11)]),
                      axis=1)]),
             axis=3)]
    ).network()

    fn = network.function(["i"], ["i1", "i2", "fu2", "fu1"])
    x = np.zeros((2, 3, 4, 5), dtype=fX)
    i1, i2, fu2, fu1 = fn(x)
    nt.assert_equal((10, 3, 4), i1.shape)
    nt.assert_equal((30, 4), i2.shape)
    nt.assert_equal((10, 3, 11), fu2.shape)
    nt.assert_equal((2, 3, 11, 5), fu1.shape)
