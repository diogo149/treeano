import numpy as np
import theano
import theano.tensor as T
import treeano.nodes as tn
fX = theano.config.floatX

# TODO change me
conv3d_node = tn.Conv3DNode
# conv3d_node = tn.DnnConv3DNode

network = tn.SequentialNode(
    "s",
    [tn.InputNode("i", shape=(1, 1, 32, 32, 32)),
     conv3d_node("conv", num_filters=32, filter_size=(3, 3, 3)),
     tn.DnnMeanPoolNode("pool", pool_size=(30, 30, 30))]
).network()
fn = network.function(["i"], ["s"])
x = np.random.randn(1, 1, 32, 32, 32).astype(fX)

"""
20150916 results:

%timeit fn(x)
Conv3DNode => 86.2 ms
DnnConv3DNode => 1.85 ms
"""
