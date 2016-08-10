import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

import treeano
from treeano.sandbox.nodes import bttf_mean

fX = theano.config.floatX


def test_backprop_to_the_future_mean_with_updates1():
    x = T.constant(treeano.utils.as_fX(0.))
    m = theano.shared(treeano.utils.as_fX(1.))
    g = theano.shared(treeano.utils.as_fX(2.))

    bm = bttf_mean.backprop_to_the_future_mean_with_updates(x, m, g, 0.5)
    fn = theano.function([], T.grad(bm, x))
    x_grad = fn()

    np.testing.assert_allclose(2.0, x_grad)
    np.testing.assert_allclose(0.5, m.get_value())
    np.testing.assert_allclose(1.5, g.get_value())


def test_backprop_to_the_future_mean_with_updates2():
    x = T.constant(treeano.utils.as_fX(0.))
    m = theano.shared(treeano.utils.as_fX(1.))
    g = theano.shared(treeano.utils.as_fX(2.))

    bm = bttf_mean.backprop_to_the_future_mean_with_updates(x, m, g, 0.7)
    fn = theano.function([], T.grad(10 * bm, x))
    x_grad = fn()

    np.testing.assert_allclose(2.0, x_grad)
    np.testing.assert_allclose(0.7, m.get_value())
    np.testing.assert_allclose(10 * 0.3 + 2 * 0.7, g.get_value())
