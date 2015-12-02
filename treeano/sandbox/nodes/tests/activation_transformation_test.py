import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import activation_transformation


fX = theano.config.floatX


def test_concatenate_negation_node_serialization():
    tn.check_serialization(
        activation_transformation.ConcatenateNegationNode("a"))


def test_concatenate_negation_node():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 10)),
         activation_transformation.ConcatenateNegationNode("a")]).network()
    fn = network.function(["i"], ["s"])
    x = np.random.randn(10, 10).astype(fX)
    ans = np.concatenate([x, -x], axis=1)
    np.testing.assert_allclose(ans, fn(x)[0])
