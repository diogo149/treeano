import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import irregular_length

fX = theano.config.floatX


def test_irregular_length_attention_softmax_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("l", shape=(None,)),
         tn.InputNode("i", shape=(None, None, 3)),
         irregular_length._IrregularLengthAttentionSoftmaxNode(
             "foo",
             lengths_reference="l")]
    ).network()

    fn = network.function(["i", "l"], ["s"])
    x = np.random.randn(4, 7, 3).astype(fX)
    l = np.array([2, 3, 7, 3], dtype=fX)
    for idx, l_ in enumerate(l):
        x[idx, l_:] = 0
    res = fn(x, l)[0]
    nt.assert_equal((4, 7, 3), res.shape)
    for idx, l_ in enumerate(l):
        np.testing.assert_almost_equal(res[idx][:l_, 0].sum(),
                                       desired=1.0,
                                       decimal=5)


def test_irregular_length_attention_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("l", shape=(None,)),
         tn.InputNode("i", shape=(None, 3)),
         irregular_length.irregular_length_attention_node(
             "foo",
             lengths_reference="l",
             num_units=3,
             output_units=None)]
    ).network()
    nt.assert_equal((None, 3), network["foo"].get_vw("default").shape)

    fn = network.function(["i", "l"], ["s"])
    x = np.random.randn(15, 3).astype(fX)
    l = np.array([2, 3, 7, 3], dtype=fX)
    res = fn(x, l)[0].shape
    ans = (4, 3)
    nt.assert_equal(ans, res)
