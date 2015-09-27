from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

fX = theano.config.floatX

if "gpu" in theano.config.device:
    import treeano.theano_extensions.fractional_max_pooling as fmp

    def test_fractional_max_pooling_numeric_gradient():
        def fun(x):
            return fmp.DisjointPseudorandomFractionalMaxPooling2DOp(
                alpha=1.414,
                u=0.5
            )(x)

        T.verify_grad(fun,
                      [np.arange(25).reshape(1, 1, 5, 5).astype(fX)],
                      rng=np.random)

    def test_fractional_max_pooling_shape():
        def fmp_shape(x, op):
            return fmp.DisjointPseudorandomFractionalMaxPooling2DOp(
                alpha=alpha,
                u=u
            )(T.constant(x)).eval().shape

        for _ in range(10):
            in_dim = np.random.randint(2, 100)
            x = np.random.randn(1, 1, in_dim, in_dim).astype(fX)
            alpha = np.random.rand() + 1
            u = np.random.rand()
            op = fmp.DisjointPseudorandomFractionalMaxPooling2DOp(
                alpha=alpha,
                u=u
            )
            res = fmp_shape(x, op)
            new_dim = op.output_length(in_dim)
            print(in_dim, res, new_dim, alpha, u)
            nt.assert_equal((1, 1, new_dim, new_dim),
                            res)
