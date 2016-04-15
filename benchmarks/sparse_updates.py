import numpy as np
import theano
import theano.tensor as T
fX = theano.config.floatX
s = theano.shared(np.zeros((30000, 10000), dtype=fX))
fn1 = theano.function([], updates=[(s, s + 1)])
fn2 = theano.function([], updates=[(s, T.inc_subtensor(s[1497], s[1497] ** 2))])

"""
on cpu:
%timeit fn1()  # 166ms
%timeit fn2()  # 14.4us
"""
