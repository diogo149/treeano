import nose.tools as nt
import numpy as np
import theano

from treeano import nodes

floatX = theano.config.floatX


@nt.raises(AssertionError)
def test_container_node_raises():
    network = nodes.SequentialNode(
        "s",
        [nodes.ContainerNode("c", []),
         nodes.IdentityNode("i")
         ]).build()
    fn = network.function([], ["i"])
    fn()


def test_splitter_node():
    network = nodes.SequentialNode(
        "s",
        [nodes.InputNode("in", shape=(3, 2, 4)),
         nodes.SplitterNode(
             "split",
             [nodes.IdentityNode("i"),
              nodes.toy.AddConstantNode("add", value=3),
              nodes.toy.MultiplyConstantNode("mult", value=2),
              ]),
         ]).build()
    x = np.random.randn(3, 2, 4).astype(floatX)
    fn1 = network.function(["in"], ["i"])
    np.testing.assert_allclose(x, fn1(x)[0])
    fn2 = network.function(["in"], ["add"])
    np.testing.assert_allclose(x + 3, fn2(x)[0])
    fn3 = network.function(["in"], ["mult"])
    np.testing.assert_allclose(x * 2, fn3(x)[0])
