import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_chunk_variables():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(None, 2)),
         tn.ApplyNode("a",
                      fn=(lambda x: x.shape[0].astype(fX) + x),
                      shape_fn=(lambda s: s))]
    ).network()

    fn1 = canopy.handlers.handled_function(
        network,
        [],
        ["i"],
        ["seq"]
    )
    np.testing.assert_equal(fn1(np.zeros((18, 2), dtype=fX)),
                            [np.ones((18, 2), dtype=fX) * 18])

    fn2 = canopy.handlers.handled_function(
        network,
        [canopy.handlers.chunk_variables(3, ["i"])],
        ["i"],
        ["seq"]
    )
    np.testing.assert_equal(fn2(np.zeros((18, 2), dtype=fX)),
                            [np.ones((18, 2), dtype=fX) * 3])
