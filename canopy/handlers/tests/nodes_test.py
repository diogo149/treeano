import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_with_hyperparameters():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.toy.AddConstantNode("ac")]
    ).network()

    fn = canopy.handlers.handled_function(
        network,
        [canopy.handlers.with_hyperparameters("hp", value=3)],
        ["i"],
        ["ac"])
    nt.assert_equal(3, fn(0)[0])


def test_override_hyperparameters():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.toy.AddConstantNode("ac", value=1)]
    ).network()

    fn = canopy.handlers.handled_function(
        network,
        [canopy.handlers.override_hyperparameters(value=2)],
        ["i"],
        ["ac"])
    nt.assert_equal(2, fn(0)[0])
