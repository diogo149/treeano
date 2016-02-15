import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

from treeano.theano_extensions.padding import pad

fX = theano.config.floatX


def test_pad():
    x = T.constant(np.array([[1, 2]], dtype=fX))
    res = pad(x, [1, 1]).eval()
    ans = np.array([[0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 0, 0, 0]], dtype=fX)
    np.testing.assert_equal(ans, res)
