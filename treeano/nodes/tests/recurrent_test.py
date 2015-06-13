import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

from treeano import nodes

fX = theano.config.floatX


def test_simple_recurrent_node_serialization():
    nodes.check_serialization(nodes.recurrent.SimpleRecurrentNode(
        "a", nodes.IdentityNode("b")))
    nodes.check_serialization(nodes.recurrent.SimpleRecurrentNode(
        "a", nodes.IdentityNode("b"), num_units=32, batch_size=2 ** 7))


def test_simple_recurrent_node():
    # just testing that it runs
    # ---
    # the test may look dumb, but it's found a LOT of problems
    network = nodes.SequentialNode(
        "n",
        [nodes.InputNode("in", shape=(3, 4, 5)),
         nodes.recurrent.SimpleRecurrentNode("srn",
                                             nodes.ReLUNode("relu"),
                                             batch_size=4,
                                             num_units=35,
                                             scan_axis=0)]
    ).network()
    fn = network.function(["in"], ["n"])
    x = np.random.rand(3, 4, 5).astype(fX)
    res = fn(x)[0]
    # 3 = scan axis, 4 = batch axis, 35 = num output units
    nt.assert_equal(res.shape, (3, 4, 35))
