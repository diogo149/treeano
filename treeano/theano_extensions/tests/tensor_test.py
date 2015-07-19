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
        fn = theano.function([var, q_var], ttt.percentile(var, q_var))
        for q in [0, 10, 42.3, 88, 100]:
            x = np.array(np.random.rand(*(dims * [4])), dtype=fX)
            ans = np.percentile(x, q)
            res = fn(x, q)
            np.testing.assert_allclose(ans, res)


def test_percentile_q_constant():
    # testing that percentile works when q is a var
    for dims, var in enumerate([T.scalar(),
                                T.vector(),
                                T.matrix(),
                                T.tensor3(),
                                T.tensor4()]):
        for q in [0, 10, 42.3, 88, 100]:
            fn = theano.function([var], ttt.percentile(var, q))
            x = np.array(np.random.rand(*(dims * [4])), dtype=fX)
            ans = np.percentile(x, q)
            res = fn(x)
            np.testing.assert_allclose(ans, res)
