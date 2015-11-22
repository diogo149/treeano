import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_monitor_variance_node_serialization():
    tn.check_serialization(tn.MonitorVarianceNode("a"))


def test_monitor_variance_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("x", shape=(3, 4, 5)),
         tn.MonitorVarianceNode("mv")]).network()
    vw = network["mv"].get_vw("var")
    x = np.random.randn(3, 4, 5).astype(fX)
    ans = x.var()
    fn = network.function(["x"], [vw.variable])
    np.testing.assert_allclose(fn(x), [ans], rtol=1e-5)
