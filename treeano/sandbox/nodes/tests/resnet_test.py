import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import resnet

fX = theano.config.floatX


def test_zero_last_axis_partition_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(None,)),
         resnet._ZeroLastAxisPartitionNode("z", zero_ratio=0.5, axis=0)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.arange(10).astype(fX)
    ans = x.copy()
    ans[5:] = 0
    np.testing.assert_allclose(ans, fn(x)[0])
