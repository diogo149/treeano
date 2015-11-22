import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import partition_axis


fX = theano.config.floatX


def test_partition_axis_node_serialization():
    tn.check_serialization(partition_axis.PartitionAxisNode("a"))


def test_partition_axis_node():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(4, 8, 12, 16, 20)),
         partition_axis.PartitionAxisNode("pa",
                                          split_idx=2,
                                          num_splits=4,
                                          channel_axis=3)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.randn(4, 8, 12, 16, 20).astype(fX)
    ans = x[:, :, :, 8:12, :]
    res = fn(x)[0]
    nt.assert_equal(ans.shape, network["pa"].get_vw("default").shape)
    np.testing.assert_equal(res, ans)
