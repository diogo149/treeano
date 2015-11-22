import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_tied_init():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=()),
         tn.AddBiasNode("b1", inits=[treeano.inits.ConstantInit(42)]),
         tn.AddBiasNode("b2", inits=[treeano.inits.TiedInit("b2", "b1")])]
    ).network()
    fn = network.function(["i"], ["s"])
    np.testing.assert_equal(84, fn(0)[0])
    network["b1"].get_vw("bias").variable.set_value(43)
    np.testing.assert_equal(86, fn(0)[0])
