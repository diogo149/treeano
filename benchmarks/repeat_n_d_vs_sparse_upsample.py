import numpy as np
import theano
import theano.tensor as T
import treeano.nodes as tn
fX = theano.config.floatX

# TODO change me
# n = tn.SpatialRepeatNDNode
n = tn.SpatialSparseUpsampleNode

network = tn.SequentialNode(
    "s",
    [tn.InputNode("i", shape=(32, 32, 32, 32, 32)),
     n("us", upsample_factor=(2, 2, 2))]
).network()
fn = network.function(["i"], ["s"])
x = np.random.randn(32, 32, 32, 32, 32).astype(fX)

"""
20150926 results:

%timeit fn(x)

SpatialRepeatNDNode: 663 ms
SpatialSparseUpsampleNode: 424 ms
"""
