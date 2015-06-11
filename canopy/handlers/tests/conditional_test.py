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

    def save_val(x):
        vals.append(x)

    network = tn.InputNode("i", shape=()).build()
    fn = canopy.handlers.handled_function(
        network,
        [canopy.handlers.call_after_every(3, save_val)],
        ["i"],
        ["i"])
    for i in range(100):
        fn(i)

    np.testing.assert_equal(np.arange(start=2, stop=100, step=3, dtype=fX),
                            np.array(vals).ravel())
