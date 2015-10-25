import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import gradient_normalization as gn

fX = theano.config.floatX


def test_gradient_batch_normalization_op():
    epsilon = 1e-8
    op = gn.GradientBatchNormalizationOp(subtract_mean=True,
                                         keep_mean=False,
                                         epsilon=epsilon)

    X = np.random.randn(3, 4).astype(fX)
    W = np.random.randn(2, 3).astype(fX)

    x = T.matrix("x")
    w = T.matrix("w")

    orig_grad = T.grad(w.dot(x).sum(), x).eval({x: X, w: W})
    new_grad = T.grad(w.dot(op(x)).sum(), x).eval({x: X, w: W})
    mu = orig_grad.mean(axis=0, keepdims=True)
    sigma = orig_grad.std(axis=0, keepdims=True) + epsilon
    ans = (orig_grad - mu) / sigma
    np.testing.assert_allclose(ans,
                               new_grad,
                               rtol=1e-5)
    np.testing.assert_allclose(np.zeros(4),
                               new_grad.mean(axis=0),
                               atol=1e-5)
    np.testing.assert_allclose(np.ones(4),
                               new_grad.std(axis=0),
                               rtol=1e-5)
