import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import prelu


fX = theano.config.floatX


def test_prelu_node_serialization():
    tn.check_serialization(prelu.PReLUNode("a"))


def test_prelu_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 4)),
         prelu.PReLUNode("p")]).network()

    fn = network.function(["i"], ["p"])
    x = np.array([[-1.0, -0.2, 0.2, 1.0]], dtype=fX)
    ans = np.array([[-0.25, -0.05, 0.2, 1.0]], dtype=fX)
    np.testing.assert_allclose(fn(x)[0],
                               ans)
