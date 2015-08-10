import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.theano_extensions.gradient as ttg

fX = theano.config.floatX


def test_gradient_reversal():
    v = np.random.randn(3, 4).astype(fX)
    m = T.matrix()
    s1 = m.sum()
    g1 = T.grad(s1, m)
    s2 = ttg.gradient_reversal(s1)
    g2 = T.grad(s2, m)
    g1_res, g2_res, s1_res, s2_res = theano.function([m], [g1, g2, s1, s2])(v)
    np.testing.assert_allclose(v.sum(), s1_res, rtol=1e-5)
    np.testing.assert_equal(s1_res, s2_res)
    np.testing.assert_equal(np.ones((3, 4), dtype=fX), g1_res)
    np.testing.assert_equal(g1_res, -g2_res)
