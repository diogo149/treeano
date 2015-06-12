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
