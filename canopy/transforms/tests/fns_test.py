import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy


fX = theano.config.floatX


def test_transform_root_node():
    network1 = tn.toy.ConstantUpdaterNode(
        "cun",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(3, 4, 5)),
             tn.LinearMappingNode(
                 "lm",
                 output_dim=15,
                 inits=[treeano.inits.NormalWeightInit(15.0)])]),
        value=-0.1,
    ).build()

    network2 = canopy.transforms.transform_root_node(network1,
                                                     fn=lambda x: x)
    network2.build()

    fn1 = network1.function(["i"], ["lm"])
    fn2 = network2.function(["i"], ["lm"])
    fn1u = network1.function(["i"], ["lm"], include_updates=True)
    fn2u = network2.function(["i"], ["lm"], include_updates=True)
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_equal(fn1(x), fn2(x))
    fn1u(x)
    np.testing.assert_equal(fn1(x), fn2(x))
    fn2u(x)
    np.testing.assert_equal(fn1(x), fn2(x))
