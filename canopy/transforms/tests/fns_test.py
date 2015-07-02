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
    ).network()
    # perform build eagerly to initialize weights
    network1.build()

    network2 = canopy.transforms.transform_root_node(network1,
                                                     fn=lambda x: x)

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


def test_transform_root_node_postwalk():
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

    def log_name(node):
        all_names.append(node.name)
        return node

    all_names = []
    canopy.transforms.transform_root_node_postwalk(network1, log_name)
    nt.assert_equal(all_names,
                    ["i", "lm", "seq", "cun"])

    def append_name(node):
        # NOTE: assumes NodeImpl subclass
        node = treeano.node_utils.copy_node(node)
        node._name += "_1"
        return node

    network2 = canopy.transforms.transform_root_node_postwalk(network1,
                                                              append_name)

    all_names = []
    canopy.transforms.transform_root_node_postwalk(network2, log_name)
    nt.assert_equal(all_names,
                    ["i_1", "lm_1", "seq_1", "cun_1"])

    # assert unmodified
    all_names = []
    canopy.transforms.transform_root_node_postwalk(network1, log_name)
    nt.assert_equal(all_names,
                    ["i", "lm", "seq", "cun"])


def test_transform_node_data_postwalk():
    network1 = tn.InputNode("i", shape=(3, 4, 5)).network()

    def change_it_up(obj):
        if obj == (3, 4, 5):
            return (6, 7, 8)
        elif obj == "i":
            return "foo"
        else:
            return obj

    network2 = canopy.transforms.transform_node_data_postwalk(network1,
                                                              change_it_up)
    x = np.random.randn(6, 7, 8).astype(fX)
    fn = network2.function(["foo"], ["foo"])
    np.testing.assert_equal(x, fn(x)[0])
