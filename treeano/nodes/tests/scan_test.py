import numpy as np
import theano
import treeano
from treeano.nodes.scan import ScanNode

floatX = theano.config.floatX


def test_basic_scan():

    class x2Node(treeano.NodeImpl):

        def compute_output(self, network, in_vw):
            network.create_vw(
                name="default",
                variable=in_vw.variable * 2,
                shape=in_vw.shape,
                tags={"output"}
            )

    network = treeano.nodes.SequentialNode(
        "seq",
        children=[
            treeano.nodes.InputNode("input", shape=(3, 2, 1)),
            ScanNode("scan", x2Node("x2"))
        ],
    ).network()
    fn = network.function(["input"], ["scan"])
    np.testing.assert_allclose(fn(np.ones((3, 2, 1)).astype(floatX))[0],
                               2 * np.ones((3, 2, 1)))
    x = np.random.rand(3, 2, 1).astype(floatX)
    np.testing.assert_allclose(fn(x)[0],
                               2 * x)
