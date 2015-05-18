from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

import treeano


def test_variable1():
    i = T.iscalar()
    o = treeano.variable.VariableWrapper("foo", variable=i).variable
    fn = theano.function([i], o)
    for _ in range(10):
        x = np.random.randint(1e6)
        assert fn(x) == x


def test_variable2():
    s = treeano.variable.VariableWrapper("foo",
                                         shape=(3, 4, 5),
                                         is_shared=True)
    assert s.value.sum() == 0
    x = np.random.randn(3, 4, 5)
    s.value = x
    assert np.allclose(s.value, x)
    try:
        s.value = np.random.randn(5, 4, 3)
    except:
        pass
    else:
        assert False
