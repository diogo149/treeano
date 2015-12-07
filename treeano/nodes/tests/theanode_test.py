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


def test_dimshuffle_node_serialization():
    tn.check_serialization(tn.DimshuffleNode("a"))


def test_gradient_reversal_node_serialization():
    tn.check_serialization(tn.GradientReversalNode("a"))


def test_zero_grad_node_serialization():
    tn.check_serialization(tn.ZeroGradNode("a"))


def test_disconnected_grad_node_serialization():
    tn.check_serialization(tn.DisconnectedGradNode("a"))


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
    assert correct_ans.shape == network["t"].get_vw("default").shape


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
                    network["ohe"].get_vw("default").shape)


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


def test_repeat_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("in", shape=(3,)),
         tn.RepeatNode("r", repeats=2, axis=0)]
    ).network()
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3).astype(fX)
    np.testing.assert_allclose(np.repeat(x, 2, 0),
                               fn(x)[0])


def test_dimshuffle_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("in", shape=(3, 4, 5)),
         tn.DimshuffleNode("r", pattern=(1, "x", 0, 2))]
    ).network()
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    ans = T.constant(x).dimshuffle(1, "x", 0, 2).eval()
    res = fn(x)[0]
    np.testing.assert_equal(res.shape, ans.shape)
    np.testing.assert_equal(res, ans)


def test_zero_grad_node():
    n1 = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.toy.ScalarSumNode("ss")]).network()

    n2 = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.ZeroGradNode("z"),
         tn.toy.ScalarSumNode("ss")]).network()

    fn1 = n1.function(["i"],
                      ["i",
                       T.grad(n1["s"].get_vw("default").variable,
                              n1["i"].get_vw("default").variable)])
    fn2 = n2.function(["i"],
                      ["i",
                       T.grad(n2["s"].get_vw("default").variable,
                              n2["i"].get_vw("default").variable)])

    # gradient should be 1 w/o zero grad node
    np.testing.assert_equal(1, fn1(3)[1])
    # gradient should be 0 w/ zero grad node
    np.testing.assert_equal(0, fn2(3)[1])


def test_disconnected_grad_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.DisconnectedGradNode("g"),
         tn.toy.ScalarSumNode("ss")]).network()

    @nt.raises(theano.gradient.DisconnectedInputError)
    def should_fail():
        T.grad(network["s"].get_vw("default").variable,
               network["i"].get_vw("default").variable)

    should_fail()
