import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_call_after_every():
    vals = []

    def save_val(in_dict, result_dict):
        vals.append(result_dict["out"])

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.call_after_every(3, save_val)],
        {"x": "i"},
        {"out": "i"})
    for i in range(100):
        fn({"x": i})

    np.testing.assert_equal(np.arange(start=2, stop=100, step=3, dtype=fX),
                            np.array(vals).ravel())
