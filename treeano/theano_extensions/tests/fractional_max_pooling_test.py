import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.theano_extensions.fractional_max_pooling as fmp

fX = theano.config.floatX


if "gpu" in theano.config.device:
    def test_fractional_max_pooling_numeric_gradient():
        def fun(x):
            return fmp.DisjointPseudorandomFractionalMaxPooling2DOp(
                alpha=1.414,
                u=0.5
            )(x)

        T.verify_grad(fun,
                      [np.arange(25).reshape(1, 1, 5, 5).astype(fX)],
                      rng=np.random)
