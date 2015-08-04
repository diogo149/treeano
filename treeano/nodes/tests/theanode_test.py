import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.nodes as tn

fX = theano.config.floatX


def test_tile_node_serialization():
    tn.check_serialization(tn.TileNode("a"))
    tn.check_serialization(tn.TileNode("a", reps=[1, 2, 3]))


def test_to_one_hot_node_serialization():
    tn.check_serialization(tn.ToOneHotNode("a"))


def test_reshape_node_serialization():
    tn.check_serialization(tn.ReshapeNode("a"))


def test_gradient_reversal_node_serialization():
    tn.check_serialization(tn.GradientReversalNode("a"))


def test_tile_node():
    network = tn.SequentialNode(
        "n",
        [tn.InputNode("in", shape=(3, 4, 5)),
         tn.TileNode("t", reps=(2, 3, 4))]
    ).network()
    fn = network.function(["in"], ["n"])
    x = np.random.rand(3, 4, 5).astype(fX)
    res = fn(x)[0]
    correct_ans = np.tile(x, (2, 3, 4))
    np.testing.assert_allclose(res, correct_ans)
    assert correct_ans.shape == network["t"].get_variable("default").shape


def test_to_one_hot_node():
    network = tn.SequentialNode(
        "n",
        [tn.InputNode("in", shape=(3,)),
         tn.ToOneHotNode("ohe", nb_class=8, cast_int32=True)]
    ).network()
    fn = network.function(["in"], ["n"])
    x = np.random.randint(8, size=3).astype(fX)
    res = fn(x)[0]
    np.testing.assert_equal(np.argmax(res, axis=1),
                            x)
    nt.assert_equal((3, 8),
                    network["ohe"].get_variable("default").shape)


def test_reshape_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("in", shape=(3, 4, 5)),
         tn.ReshapeNode("r", shape=(5, 12))]
    ).network()
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    res = fn(x)[0]
    np.testing.assert_allclose(res,
                               x.reshape(5, 12))
