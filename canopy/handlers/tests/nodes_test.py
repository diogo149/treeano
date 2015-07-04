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
         tn.AddConstantNode("ac")]
    ).network()

    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.with_hyperparameters("hp", value=3)],
        {"x": "i"},
        {"out": "ac"})
    nt.assert_equal(3, fn({"x": 0})["out"])


def test_override_hyperparameters1():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=()),
         tn.AddConstantNode("ac", value=1)]
    ).network()

    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.override_hyperparameters(value=2)],
        {"x": "i"},
        {"out": "ac"})
    nt.assert_equal(2, fn({"x": 0})["out"])


def test_override_hyperparameters2():
    network = tn.toy.ConstantUpdaterNode(
        "cun",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(3, 4, 5)),
             tn.LinearMappingNode(
                 "lm",
                 output_dim=15,
                 inits=[treeano.inits.NormalWeightInit(15.0)])]),
        value=-0.1,
    ).network()

    fn1 = network.function(["i"], ["lm"])
    fn1u = network.function(["i"], ["lm"], include_updates=True)
    fn2_args = (
        network,
        [canopy.handlers.override_hyperparameters(value=2)],
        {"x": "i"},
        {"out": "lm"}
    )
    fn2 = canopy.handlers.handled_fn(*fn2_args)
    fn2u = canopy.handlers.handled_fn(*fn2_args, include_updates=True)

    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_equal(fn1(x)[0], fn2({"x": x})["out"])
    fn1u(x)
    np.testing.assert_equal(fn1(x)[0], fn2({"x": x})["out"])
    fn2u({"x": x})
    np.testing.assert_equal(fn1(x)[0], fn2({"x": x})["out"])


def test_override_hyperparameters3():
    # testing that canopy.handlers.override_hyperparameters overrides
    # previously set override_hyperparameters
    x1 = np.array(3, dtype=fX)
    x2 = np.array(2, dtype=fX)
    network = tn.ConstantNode("c").network(
        override_hyperparameters=dict(value=x1)
    )

    fn1 = network.function([], ["c"])
    np.testing.assert_equal(fn1()[0], x1)
    fn2 = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.override_hyperparameters(value=x2)],
        {},
        {"out": "c"}
    )
    np.testing.assert_equal(fn2({})["out"], x2)
