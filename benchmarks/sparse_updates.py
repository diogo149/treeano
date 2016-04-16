import numpy as np
import theano
import theano.tensor as T
fX = theano.config.floatX
s = theano.shared(np.zeros((30000, 10000), dtype=fX))
fn1 = theano.function([], updates=[(s, s + 1)])
fn2 = theano.function([], updates=[(s, T.inc_subtensor(s[1497], s[1497] ** 2))])
# as in update deltas
fn3 = theano.function([],
                      updates=[(s, s + (T.inc_subtensor(s[1497], s[1497] ** 2) - s))])

"""
on cpu:
%timeit fn1()  # 166ms
%timeit fn2()  # 14.4us
%timeit fn3()  # 12us
"""

# doesn't work
cost = (s[0] + s[0] + 3 * s[1]).sum()
g1, g2 = theano.grad(cost, [s[0], s[1]])
# DisconnectedInputError: grad method was asked to compute the gradient
# with respect to a variable that is not part of the computational graph
# of the cost, or is used only by a non-differentiable operator:
# Subtensor{int64}.0
g1.eval()
g2.eval()


# works
s0 = s[0]
s1 = s[1]
cost = (s0 + s0 + 3 * s1).sum()
g1, g2 = theano.grad(cost, [s0, s1])
g1.eval()
g2.eval()
