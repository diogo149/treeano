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


def test_schedule_hyperparameter():
    network = tn.OutputHyperparameterNode("a", hyperparameter="foo").network(
        default_hyperparameters=dict(foo=101)
    )

    def schedule(in_dict, out_dict):
        if out_dict is None:
            return 100
        else:
            return treeano.utils.as_fX(np.random.rand() * out_dict["out"])

    fn = canopy.handled_fn(network,
                           [canopy.handlers.schedule_hyperparameter(schedule,
                                                                    "foo")],
                           {},
                           {"out": "a"})
    prev = fn({})["out"]
    assert prev != 101
    nt.assert_equal(prev, 100)
    for _ in range(10):
        curr = fn({})["out"]
        assert curr < prev
        prev = curr


def test_schedule_hyperparameter_very_leaky_relu():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.VeryLeakyReLUNode("r")]
    ).network()

    def schedule(in_dict, out_dict):
        return 10

    fn = canopy.handled_fn(
        network,
        [canopy.handlers.schedule_hyperparameter(schedule, "leak_alpha")],
        {"x": "i"},
        {"out": "s"})
    res = fn({"x": -2})["out"]
    nt.assert_equal(res, -20)


def test_use_scheduled_hyperparameter():
    network1 = tn.OutputHyperparameterNode("a", hyperparameter="foo").network(
        default_hyperparameters=dict(foo=101)
    )
    network2 = tn.SequentialNode(
        "s",
        [tn.OutputHyperparameterNode("a", hyperparameter="foo"),
         tn.MultiplyConstantNode("m", value=42)]).network(
             default_hyperparameters=dict(foo=101)
    )

    schedule = canopy.schedules.PiecewiseLinearSchedule([(1, 1), (10, 10)])
    sh_handler = canopy.handlers.schedule_hyperparameter(schedule, "foo")

    fn2 = canopy.handled_fn(
        network2,
        [canopy.handlers.use_scheduled_hyperparameter(sh_handler)],
        {},
        {"out": "s"})

    def callback(in_dict, result_dict):
        result_dict["out2"] = fn2(in_dict)["out"]

    fn1 = canopy.handled_fn(network1,
                            [sh_handler,
                             canopy.handlers.call_after_every(1, callback)],
                            {},
                            {"out": "a"})

    res = fn1({})
    nt.assert_equal(res, {"out": 1, "out2": 42})
    res = fn1({})
    nt.assert_equal(res, {"out": 2, "out2": 84})
