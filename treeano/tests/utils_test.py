import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_stable_softmax():
    x = theano.shared(np.random.randn(50, 50).astype(fX))
    s1 = T.nnet.softmax(x).eval()
    s2 = treeano.utils.stable_softmax(x).eval()
    np.testing.assert_equal(s1, s2)


def _clone_test_case(clone_fn):
    x = T.matrix("x")
    y = T.matrix("y")
    x_shape = x.shape
    sample_y = np.random.randn(4, 5).astype(theano.config.floatX)
    srng = theano.tensor.shared_randomstreams.RandomStreams()
    mask = srng.binomial(n=1, p=0.5, size=x_shape)
    mask2 = clone_fn([mask], replace={x: y})[0]
    mask2.eval({y: sample_y})  # ERROR


@nt.raises(Exception)
def test_clone():
    """
    NOTE: if this test eventually passes (eg. theano fixes the issue),
    deep_clone may no longer be necessary
    """
    _clone_test_case(theano.clone)


def test_deep_clone():
    _clone_test_case(treeano.utils.deep_clone)


def test_find_axes():
    def axes(ndim, pos, neg):
        network = tn.HyperparameterNode(
            "a",
            tn.InputNode("b", shape=()),
            pos=pos,
            neg=neg,
        ).network()["a"]
        return treeano.utils.find_axes(network, ndim, ["pos"], ["neg"])

    @nt.raises(AssertionError)
    def axes_raises(*args):
        axes(*args)

    nt.assert_equal(axes(3, [2], None), (2,))
    nt.assert_equal(axes(3, [1], None), (1,))
    nt.assert_equal(axes(3, None, [1]), (0, 2))
    nt.assert_equal(axes(3, None, [0, 1]), (2,))
    axes_raises(3, [2], [1])
