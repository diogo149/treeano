import numpy as np
import theano
import theano.tensor as T
import treeano.nodes as tn
fX = theano.config.floatX

network = tn.SequentialNode(
    "s",
    [tn.InputNode("i", shape=(32, 32, 32, 32, 32)),
     tn.SpatialRepeatNDNode("r", upsample_factor=(2, 2, 2))]
).network()
fn = network.function(["i"], ["s"])
x = np.random.randn(32, 32, 32, 32, 32).astype(fX)

"""
20150922 results:

%timeit fn(x)

from axis 0 to 4: 596 ms
from axis 4 to 0: 526 ms
"""
