import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano.core
import treeano.nodes as tn

fX = theano.config.floatX


def test_dropout_node_serialization():
    tn.check_serialization(tn.DropoutNode("a"))
    tn.check_serialization(tn.DropoutNode("a", p=0.5))


def test_gaussian_dropout_node_serialization():
    tn.check_serialization(tn.GaussianDropoutNode("a"))
    tn.check_serialization(tn.GaussianDropoutNode("a", p=0))


def test_dropout_node():
    def network(p):
        return tn.SequentialNode("s", [
            tn.InputNode("i", shape=(3, 4, 5)),
            tn.DropoutNode("do", p=p)
        ]).network()

    x = np.random.randn(3, 4, 5).astype(fX)
    fn1 = network(0).function(["i"], ["s"])
    np.testing.assert_allclose(fn1(x)[0], x)

    @nt.raises(AssertionError)
    def tmp():
        fn2 = network(0.5).function(["i"], ["s"])
        np.testing.assert_allclose(fn2(x)[0], x)

    tmp()


def test_gaussian_dropout_node():
    def network(p):
        return tn.SequentialNode("s", [
            tn.InputNode("i", shape=(3, 4, 5)),
            tn.GaussianDropoutNode("do", p=p)
        ]).network()

    x = np.random.randn(3, 4, 5).astype(fX)
    fn1 = network(0).function(["i"], ["s"])
    np.testing.assert_allclose(fn1(x)[0], x)

    @nt.raises(AssertionError)
    def tmp():
        fn2 = network(0.5).function(["i"], ["s"])
        np.testing.assert_allclose(fn2(x)[0], x)

    tmp()
