import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import inverse

fX = theano.config.floatX


def test_inverse_node_serialization():
    tn.check_serialization(inverse.InverseNode("a"))


def test_inverse_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 1, 2, 2)),
         tn.MaxPool2DNode("m", pool_size=(2, 2)),
         tn.InputNode("i2", shape=(1, 1, 1, 1)),
         inverse.InverseNode("in", reference="m")]
    ).network()
    fn = network.function(["i", "i2"], ["in"])
    x = np.array([[[[1, 2],
                    [3, 4]]]],
                 dtype=fX)
    x2 = np.array(np.random.randn(), dtype=fX)
    ans = x2 * np.array([[[[0, 0],
                           [0, 1]]]],
                        dtype=fX)

    np.testing.assert_equal(ans, fn(x, x2.reshape(1, 1, 1, 1))[0])
