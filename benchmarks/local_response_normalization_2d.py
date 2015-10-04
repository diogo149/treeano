import numpy as np
import theano
import theano.tensor as T
import treeano
from treeano.sandbox.nodes import lrn
fX = theano.config.floatX

# shape = (3, 4, 5, 6)
# shape = (32, 32, 32, 32)
shape = (128, 32, 128, 128)

f = lrn.local_response_normalization_2d_v1
# f = lrn.local_response_normalization_2d_v2
# f = lrn.local_response_normalization_2d_dnn
# f = lrn.local_response_normalization_2d_pool

vw = treeano.VariableWrapper("foo",
                             variable=T.tensor4(),
                             shape=shape)
kwargs = dict(
    alpha=1e-4,
    k=2,
    beta=0.75,
    n=5,
)
target = f(vw, **kwargs).sum()
g_sum = T.grad(target, vw.variable).sum()
fn1 = theano.function(
    [vw.variable],
    [target])
fn2 = theano.function(
    [vw.variable],
    [g_sum])
x = np.random.randn(*shape).astype(fX)

"""
20151004 results:

=====

shape = (3, 4, 5, 6)

forward pass:
%timeit fn1(x)
local_response_normalization_2d_v1: 66.2 us
local_response_normalization_2d_v2: 66.5 us
local_response_normalization_2d_dnn: 61.8 us
local_response_normalization_2d_pool: 66.6 us

forward + backward pass:
%timeit fn2(x)
local_response_normalization_2d_v1: 117 us
local_response_normalization_2d_v2: 115 us
local_response_normalization_2d_dnn: 91.4 us
local_response_normalization_2d_pool: 87.4 us

=====

shape = (32, 32, 32, 32)

forward pass:
%timeit fn1(x)
local_response_normalization_2d_v1: 2.26 ms
local_response_normalization_2d_v2: 2.29 ms
local_response_normalization_2d_pool: 2.15 ms

forward + backward pass:
%timeit fn2(x)
local_response_normalization_2d_v1: 6.71 ms
local_response_normalization_2d_v2: 6.71 ms
local_response_normalization_2d_pool: 2.69 ms

=====

shape = (128, 32, 128, 128)

forward pass:
%timeit fn1(x)
local_response_normalization_2d_v1: 145 ms
local_response_normalization_2d_pool: 139 ms

forward + backward pass:
%timeit fn2(x)
local_response_normalization_2d_v1: 584 ms
local_response_normalization_2d_pool: 167 ms
"""
