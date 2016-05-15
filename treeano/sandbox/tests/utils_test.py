import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox import utils

fX = theano.config.floatX


def test_overwrite_grad_multiple_args():
    class Foo(utils.OverwriteGrad):

        def __init__(self):
            def fn(a, b):
                return a + 2 * b

            super(Foo, self).__init__(fn)

        def grad(self, inputs, out_grads):
            a, b = inputs
            grd, = out_grads
            return a, a * b * grd

    foo_op = Foo()

    a = T.scalar()
    b = T.scalar()
    ga, gb = T.grad(3 * foo_op(a, b), [a, b])
    fn = theano.function([a, b], [ga, gb])
    res1, res2 = fn(2.7, 11.4)
    np.testing.assert_allclose(res1, 2.7)
    np.testing.assert_allclose(res2, 2.7 * 11.4 * 3)
