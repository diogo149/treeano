import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_callback_with_input():
    vals = []

    def save_sum(in_dict, result_dict):
        vals.append(result_dict["out"] + in_dict["x"])

    network = tn.InputNode("i", shape=()).network()
    fn = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.callback_with_input(save_sum)],
        {"x": "i"},
        {"out": "i"})
    for i in range(100):
        fn({"x": i})

    np.testing.assert_equal(2 * np.arange(100),
                            np.array(vals).ravel())
