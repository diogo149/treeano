import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano


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
