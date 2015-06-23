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
