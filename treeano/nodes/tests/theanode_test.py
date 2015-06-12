import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

from treeano import nodes

fX = theano.config.floatX


def test_tile_node_serialization():
    nodes.check_serialization(nodes.TileNode("a"))
    nodes.check_serialization(nodes.TileNode("a", reps=[1, 2, 3]))


def test_tile_node():
    network = nodes.SequentialNode(
        "n",
        [nodes.InputNode("in", shape=(3, 4, 5)),
         nodes.TileNode("t", reps=(2, 3, 4))]
    ).build()
    fn = network.function(["in"], ["n"])
    x = np.random.rand(3, 4, 5).astype(fX)
    res = fn(x)[0]
    correct_ans = np.tile(x, (2, 3, 4))
    np.testing.assert_allclose(res, correct_ans)
    assert correct_ans.shape == network["t"].get_variable("default").shape
