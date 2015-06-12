import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

import canopy

fX = theano.config.floatX


def test_to_shared_dict():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(10,)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=[treeano.inits.ConstantInit(42.42)])]
    ).build()
    sd = canopy.network_utils.to_shared_dict(network)
    nt.assert_equal(sd.keys(), ["lm:weight"])
    np.testing.assert_equal(sd.values()[0].get_value(),
                            42.42 * np.ones((10, 15), dtype=fX))


def test_to_preallocated_init():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=[treeano.inits.NormalWeightInit(15.0)])]
    ).build()
    inits = [canopy.network_utils.to_preallocated_init(network1)]
    network2 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=inits)]
    ).build()

    w1 = canopy.network_utils.to_shared_dict(network1).values()[0]
    w2 = canopy.network_utils.to_shared_dict(network2).values()[0]
    # both networks should be using the exact same shared variables
    assert w1 is w2

    fn1 = network1.function(["i"], ["lm"])
    fn2 = network2.function(["i"], ["lm"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_equal(fn1(x),
                            fn2(x))
