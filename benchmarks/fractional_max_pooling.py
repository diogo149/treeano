import numpy as np
import theano
import theano.tensor as T
import treeano.nodes as tn
from treeano.sandbox.nodes import fmp
fX = theano.config.floatX

# TODO change me
node = "fmp2"
compute_grad = True

if node == "mp":
    n = tn.MaxPool2DNode("mp", pool_size=(2, 2))
elif node == "fmp":
    n = fmp.DisjointPseudorandomFractionalMaxPool2DNode("fmp1",
                                                        fmp_alpha=1.414,
                                                        fmp_u=0.5)
elif node == "fmp2":
    n = fmp.OverlappingRandomFractionalMaxPool2DNode("fmp2",
                                                     pool_size=(1.414, 1.414))
else:
    assert False


network = tn.SequentialNode(
    "s",
    [tn.InputNode("i", shape=(1, 1, 32, 32)),
     n]
).network()


if compute_grad:
    i = network["i"].get_vw("default").variable
    s = network["s"].get_vw("default").variable
    fn = network.function(["i"], [T.grad(s.sum(), i)])
else:
    fn = network.function(["i"], ["s"])

x = np.random.randn(1, 1, 32, 32).astype(fX)

"""
20150924 results:

%timeit fn(x)

no grad:
mp: 33.7 us
fmp: 77.6 us
fmp2: 1.91 ms

with grad:
mp: 67.1 us
fmp: 162 us
fmp2: 2.66 ms
"""
