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

    fn1 = canopy.handlers.handled_fn(network,
                                     [],
                                     {"x": "i"},
                                     {"out": "seq"})
    np.testing.assert_equal(fn1({"x": np.zeros((18, 2), dtype=fX)})["out"],
                            np.ones((18, 2), dtype=fX) * 18)

    fn2 = canopy.handlers.handled_fn(
        network,
        [canopy.handlers.chunk_variables(3, ["i"])],
        {"x": "i"},
        {"out": "seq"})
    np.testing.assert_equal(fn2({"x": np.zeros((18, 2), dtype=fX)})["out"],
                            np.ones((18, 2), dtype=fX) * 3)


def test_batch_pad():

    def tmp(include_batch_pad):
        network = tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(None, 2)),
             tn.ApplyNode("a",
                          fn=(lambda x: x.shape[0].astype(fX) + x),
                          shape_fn=(lambda s: s))]
        ).network()
        handlers = [canopy.handlers.chunk_variables(3, ["i"])]
        if include_batch_pad:
            handlers.insert(0, canopy.handlers.batch_pad(3, ["x"]))
        fn = canopy.handlers.handled_fn(network,
                                        handlers,
                                        {"x": "i"},
                                        {"out": "seq"})
        return fn({"x": np.zeros((16, 2), dtype=fX)})

    nt.raises(AssertionError)(tmp)(False)
    res = tmp(True)

    np.testing.assert_equal(res["out"], np.ones((18, 2), dtype=fX) * 3)
