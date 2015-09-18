import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy


fX = theano.config.floatX


def test_remove_dropout():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.DropoutNode("do", dropout_probability=0.5)]).network()
    network2 = canopy.transforms.remove_dropout(network1)

    assert "DropoutNode" in str(network1.root_node)
    assert "DropoutNode" not in str(network2.root_node)

    fn1 = network1.function(["i"], ["do"])
    fn2 = network2.function(["i"], ["do"])
    x = np.random.randn(3, 4, 5).astype(fX)

    @nt.raises(AssertionError)
    def fails():
        np.testing.assert_equal(x, fn1(x)[0])

    fails()
    np.testing.assert_equal(x, fn2(x)[0])


def test_replace_node():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.DropoutNode("do", dropout_probability=0.5)]).network()
    network2 = canopy.transforms.replace_node(network1,
                                              {"do": tn.IdentityNode("do")})

    assert "DropoutNode" in str(network1.root_node)
    assert "DropoutNode" not in str(network2.root_node)

    fn1 = network1.function(["i"], ["do"])
    fn2 = network2.function(["i"], ["do"])
    x = np.random.randn(3, 4, 5).astype(fX)

    @nt.raises(AssertionError)
    def fails():
        np.testing.assert_equal(x, fn1(x)[0])

    fails()
    np.testing.assert_equal(x, fn2(x)[0])


def test_update_hyperparameters():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.DropoutNode("do", dropout_probability=0.5)]).network()
    network2 = canopy.transforms.update_hyperparameters(
        network1,
        "do",
        {"dropout_probability": 0.3})

    assert network1["do"].find_hyperparameter(["dropout_probability"]) == 0.5
    assert network2["do"].find_hyperparameter(["dropout_probability"]) == 0.3
