import numpy as np
import theano
import theano.tensor as T
import treeano.nodes as tn
fX = theano.config.floatX


def elu1(x, alpha=1.):
    return T.switch(T.gt(x, 0.), x, alpha * (T.exp(x) - 1))


def elu2(x, alpha=1.):
    pos = (x + abs(x)) / 2
    neg = (x + -abs(x)) / 2
    return pos + alpha * (T.exp(neg) - 1)


for _ in range(100):
    tmp = np.random.randn()
    np.testing.assert_allclose(elu1(tmp).eval(),
                               elu2(tmp).eval())

# TODO change me
# elu = elu1
elu = elu2

x = T.matrix()
f = elu(x)
b = T.grad(f.sum(), x)
X = np.random.randn(4096, 4096).astype(fX)

"""
20151204 results

%timeit f.eval({x: X})
elu1 => 33.3 ms
elu2 => 28.3 ms

%timeit b.eval({x: X})
elu1 => 161 ms
elu2 => 29.2 ms
"""
