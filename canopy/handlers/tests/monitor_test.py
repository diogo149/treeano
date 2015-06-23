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
        [canopy.handlers.time_call()],
        {"x": "i"},
        {"out": "i"})
    res = fn({"x": 0})
    assert "time" in res
    assert "out" in res
