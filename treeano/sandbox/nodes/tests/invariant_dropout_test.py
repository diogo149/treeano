import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import invariant_dropout as ido


fX = theano.config.floatX


def test_invariant_dropout_node_serialization():
    tn.check_serialization(ido.InvariantDropoutNode("a"))


def test_invariant_dropout_node():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 10)),
         ido.InvariantDropoutNode("ido", p=0.5)]).network()

    fn = network.function(["i"], ["s"])
    x = np.random.rand(10, 10).astype(fX)
    fn(x)
