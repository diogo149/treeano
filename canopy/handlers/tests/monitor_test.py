import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_time_call():
    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.time_call(key="time")],
        {"x": "i"},
        {"out": "i"})
    res = fn({"x": 0})
    assert "time" in res
    assert "out" in res


def test_time_per_row1():
    network = tn.InputNode("i", shape=(10,)).network()
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.time_per_row(input_key="x", key="ms_per_row")],
        {"x": "i"},
        {"out": "i"})
    res = fn({"x": np.random.rand(10).astype(fX)})
    assert "ms_per_row" in res
    assert "out" in res


@nt.raises(Exception)
def test_time_per_row2():
    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.time_per_row(input_key="x", key="ms_per_row")],
        {"x": "i"},
        {"out": "i"})
    fn({"x": 0})


def test_evaluate_monitoring_variables():

    class FooNode(treeano.NodeImpl):

        def compute_output(self, network, in_vw):
            network.create_variable(
                "default",
                variable=42 * in_vw.variable.sum(),
                shape=(),
                tags={"monitor"}
            )

    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(3, 4, 5)),
         FooNode("f")]
    ).network()
    x = np.random.randn(3, 4, 5).astype(fX)
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.evaluate_monitoring_variables(fmt="train_%s")],
        {"x": "i"},
        {})
    res = fn({"x": x})
    ans_key = "train_f:default"
    assert ans_key in res
    np.testing.assert_allclose(res[ans_key], 42 * x.sum(), rtol=1e-5)


def test_output_nanguard():
    def output_nanguard_fn(a):
        class CustomNode(treeano.NodeImpl):

            def compute_output(self, network, in_vw):
                network.create_variable(
                    "default",
                    variable=in_vw.variable / a,
                    shape=in_vw.shape
                )

        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=()),
             CustomNode("c")]
        ).network()

        return canopy.handlers.handled_fn(
            network,
            [canopy.handlers.output_nanguard()],
            {"x": "i"},
            {"out": "s"})

    fn1 = output_nanguard_fn(3)
    np.testing.assert_equal(fn1({"x": 3}), {"out": np.array(1)})
    np.testing.assert_equal(fn1({"x": -6}), {"out": np.array(-2)})

    fn2 = output_nanguard_fn(0)
    try:
        fn2({"x": 3})
    except Exception as e:
        assert e.message["error_type"] == "inf"
        np.testing.assert_equal(e.message["value"], np.array(np.inf))
    else:
        assert False

    try:
        fn2({"x": -6})
    except Exception as e:
        nt.assert_equal(e.message["error_type"], "inf")
        np.testing.assert_equal(e.message["value"], np.array(-np.inf))
    else:
        assert False

    try:
        fn2({"x": 0})
    except Exception as e:
        nt.assert_equal(e.message["error_type"], "nan")
        np.testing.assert_equal(e.message["value"], np.array(np.nan))
    else:
        assert False

    try:
        fn1({"x": 6e10})
    except Exception as e:
        nt.assert_equal(e.message["error_type"], "big")
        np.testing.assert_allclose(e.message["value"],
                                   np.array(2e10),
                                   rtol=1e-5)
    else:
        assert False


def test_nanguardmode():
    def nanguardmode_fn(a):
        class CustomNode(treeano.NodeImpl):

            def compute_output(self, network, in_vw):
                network.create_variable(
                    "default",
                    variable=in_vw.variable / a,
                    shape=in_vw.shape
                )

        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=()),
             CustomNode("c")]
        ).network()

        return canopy.handlers.handled_fn(
            network,
            [canopy.handlers.nanguardmode()],
            {"x": "i"},
            {"out": "s"})

    fn1 = nanguardmode_fn(3)
    np.testing.assert_equal(fn1({"x": 3}), {"out": np.array(1)})
    np.testing.assert_equal(fn1({"x": -6}), {"out": np.array(-2)})

    @nt.raises(AssertionError)
    def raises_fn1(x):
        fn1({"x": x})

    raises_fn1(6e10)

    fn2 = nanguardmode_fn(0)

    @nt.raises(AssertionError)
    def raises_fn2(x):
        fn2({"x": x})

    raises_fn2(3)
    raises_fn2(-6)
    raises_fn2(0)
    raises_fn2(6e10)
