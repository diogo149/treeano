import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import kumaraswamy_unit as ku


fX = theano.config.floatX


def test_kumaraswamy_unit_node_serialization():
    tn.check_serialization(ku.KumaraswamyUnitNode("a"))


def test_kumaraswamy_unit_node():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(100,)),
         ku.KumaraswamyUnitNode("k")]).network()
    fn = network.function(["i"], ["s"])
    x = np.random.randn(100).astype(fX)
    fn(x)
