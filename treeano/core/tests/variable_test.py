from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

import treeano

fX = theano.config.floatX


def test_variable1():
    i = T.iscalar()
    o = treeano.core.variable.VariableWrapper("foo", variable=i).variable
    fn = theano.function([i], o)
    for _ in range(10):
        x = np.random.randint(1e6)
        assert fn(x) == x


def test_variable2():
    s = treeano.core.variable.VariableWrapper("foo",
                                              shape=(3, 4, 5),
                                              is_shared=True,
                                              inits=[])
    assert s.value.sum() == 0
    x = np.random.randn(3, 4, 5).astype(theano.config.floatX)
    s.value = x
    assert np.allclose(s.value, x)
    try:
        s.value = np.random.randn(5, 4, 3)
    except:
        pass
    else:
        assert False


def test_variable_symbolic_shape():
    m = T.matrix()
    f = treeano.core.variable.VariableWrapper("foo",
                                              variable=m,
                                              shape=(4, None))
    s = f.symbolic_shape()
    assert isinstance(s, tuple)
    assert s[0] == 4
    assert isinstance(s[1], theano.gof.graph.Variable)
    assert s[1].eval({m: np.zeros((4, 100), dtype=fX)}) == 100
