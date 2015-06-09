import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_concatenate_node_serialization():
    tn.check_serialization(tn.ConcatenateNode("a", []))
    tn.check_serialization(tn.ConcatenateNode(
        "a",
        [tn.ConcatenateNode("b", []),
         tn.ConcatenateNode("c", [])]))


def test_concatenate_node():

    def replace_nones(shape):
        return [s if s is not None else np.random.randint(1, 100)
                for s in shape]

    for axis, s1, s2, s3 in [[0] + [(3, 2, 4)] * 3,
                             [0, (3, 2, 4), (5, 2, 4), (6, 2, 4)],
                             [0, (3, 2, 4), (5, 2, 4), (None, 2, 4)],
                             [1, (3, 5, 4), (3, 1, 4), (3, 2, 4)], ]:
        network = tn.ConcatenateNode(
            "concat",
            [tn.InputNode("i1", shape=s1),
             tn.InputNode("i2", shape=s2),
             tn.InputNode("i3", shape=s3)],
            axis=axis,
        ).build()
        fn = network.function(["i1", "i2", "i3"], ["concat"])
        i1 = np.random.rand(*replace_nones(s1)).astype(fX)
        i2 = np.random.rand(*replace_nones(s2)).astype(fX)
        i3 = np.random.rand(*replace_nones(s3)).astype(fX)
        np.testing.assert_allclose(np.concatenate([i1, i2, i3], axis=axis),
                                   fn(i1, i2, i3)[0])


def test_concatenate_node_wrong_shape():

    @nt.raises(AssertionError)
    def build_concatenate(axis, s1, s2, s3):
        tn.ConcatenateNode(
            "concat",
            [tn.InputNode("i1", shape=s1),
             tn.InputNode("i2", shape=s2),
             tn.InputNode("i3", shape=s3)],
            axis=axis,
        ).build()

    for args in [[100] + [(3, 2, 4)] * 3,
                 [0, (3, 2, 4), (3, 2, 4), (3, 3, 4)],
                 [0, (3, 2, 4), (3, 2, 4), (3, None, 4)], ]:
        build_concatenate(*args)
