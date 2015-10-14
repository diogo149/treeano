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
    ).network()
    sd = canopy.network_utils.to_shared_dict(network)
    nt.assert_equal(["lm:weight"], list(sd.keys()))
    np.testing.assert_equal(42.42 * np.ones((10, 15), dtype=fX),
                            list(sd.values())[0].get_value())


def test_to_shared_dict_relative_network():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(10,)),
         tn.LinearMappingNode("lm1", output_dim=15),
         tn.LinearMappingNode("lm2", output_dim=15)]
    ).network()
    nt.assert_equal({"lm1:weight", "lm2:weight"},
                    set(canopy.network_utils.to_shared_dict(network)))
    nt.assert_equal({"lm1:weight"},
                    set(canopy.network_utils.to_shared_dict(network["lm1"])))
    nt.assert_equal({"lm2:weight"},
                    set(canopy.network_utils.to_shared_dict(network["lm2"])))


def test_to_value_dict():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(10,)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=[treeano.inits.ConstantInit(42.42)])]
    ).network()
    sd = canopy.network_utils.to_value_dict(network)
    nt.assert_equal(["lm:weight"], list(sd.keys()))
    np.testing.assert_equal(42.42 * np.ones((10, 15), dtype=fX),
                            sd["lm:weight"])


def test_load_value_dict():
    def new_network():
        return tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(10, 100)),
             tn.LinearMappingNode(
                 "lm",
                 output_dim=15,
                 inits=[treeano.inits.NormalWeightInit()])]
        ).network()

    n1 = new_network()
    n2 = new_network()

    fn1 = n1.function(["i"], ["lm"])
    fn2 = n2.function(["i"], ["lm"])

    x = np.random.randn(10, 100).astype(fX)

    def test():
        np.testing.assert_equal(fn1(x), fn2(x))

    # should fail
    nt.raises(AssertionError)(test)()
    # change weights
    canopy.network_utils.load_value_dict(
        n1, canopy.network_utils.to_value_dict(n2))
    # should not fail
    test()


def test_load_value_dict_not_strict_keys():
    n1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(10, 100)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=[treeano.inits.NormalWeightInit()])]
    ).network()
    n2 = tn.InputNode("i", shape=()).network()

    def test1(strict_keys):
        canopy.network_utils.load_value_dict(
            n1,
            canopy.network_utils.to_value_dict(n2),
            strict_keys=strict_keys)

    def test2(strict_keys):
        canopy.network_utils.load_value_dict(
            n2,
            canopy.network_utils.to_value_dict(n1),
            strict_keys=strict_keys)

    nt.raises(AssertionError)(test1)(strict_keys=True)
    nt.raises(AssertionError)(test2)(strict_keys=True)
    test1(strict_keys=False)
    test2(strict_keys=False)


def test_to_preallocated_init1():
    network1 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=[treeano.inits.NormalWeightInit(15.0)])]
    ).network()
    inits = [canopy.network_utils.to_preallocated_init(network1)]
    network2 = tn.SequentialNode(
        "seq",
        [tn.InputNode("i", shape=(3, 4, 5)),
         tn.LinearMappingNode(
             "lm",
             output_dim=15,
             inits=inits)]
    ).network()

    w1 = list(canopy.network_utils.to_shared_dict(network1).values())[0]
    w2 = list(canopy.network_utils.to_shared_dict(network2).values())[0]
    # both networks should be using the exact same shared variables
    assert w1 is w2

    fn1 = network1.function(["i"], ["lm"])
    fn2 = network2.function(["i"], ["lm"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_equal(fn1(x),
                            fn2(x))


def test_to_preallocated_init2():
    # test that networks are kept in sync even when updating
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
    ).network()
    inits = [canopy.network_utils.to_preallocated_init(network1)]
    network2 = tn.toy.ConstantUpdaterNode(
        "cun",
        tn.SequentialNode(
            "seq",
            [tn.InputNode("i", shape=(3, 4, 5)),
             tn.LinearMappingNode(
                 "lm",
                 output_dim=15,
                 inits=inits)]),
        value=0.4,
    ).network()

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
