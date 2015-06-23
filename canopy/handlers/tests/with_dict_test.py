import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


fX = theano.config.floatX


def test_call_with_dict():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(None, 2)),
         tn.ApplyNode("a",
                      fn=(lambda x: x + 42),
                      shape_fn=(lambda s: s))]
    ).network()

    fn = canopy.handlers.handled_function(
        network,
        [canopy.handlers.call_with_dict()],
        {"foo": ("i", "default")},
        ["seq"]
    )
    np.testing.assert_equal(fn({"foo": np.zeros((18, 2), dtype=fX)}),
                            [np.ones((18, 2), dtype=fX) * 42])


def test_return_dict():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(None, 2)),
         tn.ApplyNode("a",
                      fn=(lambda x: x + 42),
                      shape_fn=(lambda s: s))]
    ).network()

    fn = canopy.handlers.handled_function(
        network,
        [canopy.handlers.return_dict()],
        ["i"],
        {"foo": "seq"},
    )
    np.testing.assert_equal(fn(np.zeros((18, 2), dtype=fX)),
                            {"foo": np.ones((18, 2), dtype=fX) * 42})
