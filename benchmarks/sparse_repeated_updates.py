import numpy as np
import theano
import theano.tensor as T
fX = theano.config.floatX
s = theano.shared(np.ones((10, 1), dtype=fX))
idxs = [0, 1, 1]
fn = theano.function([], updates=[(s, T.inc_subtensor(s[idxs], s[idxs] ** 2))])
fn()
print s.get_value()
