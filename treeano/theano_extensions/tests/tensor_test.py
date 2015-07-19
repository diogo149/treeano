import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.theano_extensions.tensor as ttt

fX = theano.config.floatX


def test_percentile():
    q_var = T.scalar()
    for dims, var in enumerate([T.scalar(),
                                T.vector(),
                                T.matrix(),
                                T.tensor3(),
                                T.tensor4()]):
        fn = theano.function([var, q_var], [ttt.percentile(var, q_var)])
        for q in [0, 10, 42.3, 88, 100]:
            x = np.array(np.random.rand(*(dims * [4])), dtype=fX)
            ans = np.percentile(x, q)
            res = fn(x, q)[0]
            np.testing.assert_allclose(ans, res)


def test_percentile_q_constant():
    # testing that percentile works when q is not a var
    for dims, var in enumerate([T.scalar(),
                                T.vector(),
                                T.matrix(),
                                T.tensor3(),
                                T.tensor4()]):
        for q in [0, 10, 42.3, 88, 100]:
            fn = theano.function([var], [ttt.percentile(var, q)])
            x = np.array(np.random.rand(*(dims * [4])), dtype=fX)
            ans = np.percentile(x, q)
            res = fn(x)[0]
            np.testing.assert_allclose(ans, res)


def test_percentile_axis():
    v = T.tensor4()
    x = np.array(np.random.rand(*range(4, 8)), dtype=fX)
    q = 42.42
    for axis in [None,
                 [2, 3],
                 [0, 1],
                 [0, 1, 2, 3],
                 3]:
        fn = theano.function([v], [ttt.percentile(v, q, axis=axis)])
        ans = np.percentile(x, q, axis=axis)
        res = fn(x)[0]
        np.testing.assert_allclose(ans, res, rtol=1e-5)


def test_percentile_keepdims():
    v = T.tensor4()
    x = np.array(np.random.rand(*range(4, 8)), dtype=fX)
    q = 42.42
    for keepdims in [True, False]:
        for axis in [None,
                     [2, 3],
                     [0, 1],
                     [0, 1, 2, 3],
                     3]:
            fn = theano.function([v],
                                 [ttt.percentile(v,
                                                 q,
                                                 axis=axis,
                                                 keepdims=keepdims)])
            ans = np.percentile(x, q, axis=axis, keepdims=keepdims)
            res = fn(x)[0]
            np.testing.assert_allclose(ans, res, rtol=1e-5)


def test_percentile_keepdims_broadcastable():
    v = T.tensor4()
    x = np.array(np.random.rand(*range(4, 8)), dtype=fX)
    q = 42.42
    ans = x + np.percentile(x, q, axis=[2, 3], keepdims=True)
    res_var = v + ttt.percentile(v,
                                 q,
                                 axis=[2, 3],
                                 keepdims=True)
    fn = theano.function([v], [res_var])
    res = fn(x)[0]
    np.testing.assert_allclose(ans, res, rtol=1e-5)


def test_percentile_keepdims_input_broadcastable():
    v = T.tensor4()
    nt.assert_equal(ttt.percentile(v.dimshuffle(0, 1, 2, "x", 3),
                                   0,
                                   axis=[2]).broadcastable,
                    (False, False, True, False))
    v = T.tensor4()
    nt.assert_equal(ttt.percentile(v.dimshuffle(0, 1, 2, 3, "x"),
                                   0,
                                   axis=[2],
                                   keepdims=True).broadcastable,
                    (False, False, True, False, True))


# TODO implement gradient
# def test_percentile_grad():
#     # test that gradient works and returns 0
#     v = T.tensor4()
#     g = T.grad(ttt.percentile(v, 32.42), v)
#     fn = theano.function([v], [g])
#     x = np.array(np.random.rand(*range(4, 8)), dtype=fX)
#     res = fn(x)[0]
#     np.testing.assert_equal(res,
#                             np.zeros_like(x, dtype=fX))
