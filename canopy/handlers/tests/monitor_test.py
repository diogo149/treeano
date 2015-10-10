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
            network.create_vw(
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


def test_monitor_network_state():

    class CustomNode(treeano.NodeImpl):
        input_keys = ()

        def compute_output(self, network):
            network.create_vw(
                "default",
                is_shared=True,
                shape=(),
                inits=[]
            )

    network = CustomNode("c").network()
    # build eagerly to share weights
    network.build()

    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.monitor_network_state()],
        {},
        {})

    res = fn({})
    # assert that there is a key that has "mean" in it
    assert any("mean" in k for k in res.keys())
    # assert that there is a key that has "std" in it
    assert any("std" in k for k in res.keys())
