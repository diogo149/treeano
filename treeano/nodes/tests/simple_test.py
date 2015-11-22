import copy

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_reference_node_serialization():
    tn.check_serialization(tn.ReferenceNode("a"))
    tn.check_serialization(tn.ReferenceNode("a", reference="bar"))


def test_send_to_node_serialization():
    tn.check_serialization(tn.SendToNode("a"))
    tn.check_serialization(tn.SendToNode("a", reference="bar"))


def test_hyperparameter_node_serialization():
    tn.check_serialization(
        tn.HyperparameterNode("a",
                              tn.ReferenceNode("b")))


def test_add_bias_node_serialization():
    tn.check_serialization(tn.AddBiasNode("a"))
    tn.check_serialization(tn.AddBiasNode(
        "a",
        inits=[],
        # need to make broadcastable a list because json (de)serialization
        # converts tuples to lists
        broadcastable=[True, False, True]))


def test_linear_mapping_node_serialization():
    tn.check_serialization(tn.LinearMappingNode("a"))
    tn.check_serialization(tn.LinearMappingNode("a", output_dim=3))


def test_apply_node_serialization():
    tn.check_serialization(tn.ApplyNode("a"))


def test_reference_node():
    network = tn.SequentialNode("s", [
        tn.InputNode("input1", shape=(3, 4, 5)),
        tn.InputNode("input2", shape=(5, 4, 3)),
        tn.ReferenceNode("ref", reference="input1"),
    ]).network()

    fn = network.function(["input1"], ["ref"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_allclose(fn(x)[0], x)


def test_send_to_node():
    network = tn.ContainerNode("c", [
        tn.SequentialNode(
            "s1",
            [tn.InputNode("in", shape=(3, 4, 5)),
             tn.SendToNode("stn1", reference="s2")]),
        tn.SequentialNode(
            "s2",
            [tn.SendToNode("stn2", reference="stn3")]),
        tn.SequentialNode(
            "s3",
            [tn.SendToNode("stn3", reference="i")]),
        tn.IdentityNode("i"),
    ]).network()

    fn = network.function(["in"], ["i"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_allclose(fn(x)[0], x)


def test_network_doesnt_mutate():
    root_node = tn.ContainerNode("c", [
        tn.SequentialNode(
            "s1",
            [tn.InputNode("in", shape=(3, 4, 5)),
             tn.SendToNode("stn1", reference="s2")]),
        tn.SequentialNode(
            "s2",
            [tn.SendToNode("stn2", reference="stn3")]),
        tn.SequentialNode(
            "s3",
            [tn.SendToNode("stn3", reference="i")]),
        tn.IdentityNode("i"),
    ])
    original_dict = copy.deepcopy(root_node.__dict__)
    root_node.network().build()
    nt.assert_equal(original_dict,
                    root_node.__dict__)


def test_node_with_generated_children_can_serialize():
    root_node = tn.ContainerNode("c", [
        tn.SequentialNode(
            "s1",
            [tn.InputNode("in", shape=(3, 4, 5)),
             tn.SendToNode("stn1", reference="s2")]),
        tn.SequentialNode(
            "s2",
            [tn.SendToNode("stn2", reference="stn3")]),
        tn.SequentialNode(
            "s3",
            [tn.SendToNode("stn3", reference="i")]),
        tn.IdentityNode("i"),
    ])
    root_node.network().build()
    root2 = treeano.core.node_from_data(treeano.core.node_to_data(root_node))
    nt.assert_equal(root_node, root2)


def test_add_bias_node_broadcastable():
    def get_bias_shape(broadcastable):
        return tn.SequentialNode("s", [
            tn.InputNode("in", shape=(3, 4, 5)),
            (tn.AddBiasNode("b", broadcastable=broadcastable)
             if broadcastable is not None
             else tn.AddBiasNode("b"))
        ]).network()["b"].get_vw("bias").shape

    nt.assert_equal((1, 4, 5),
                    get_bias_shape(None))
    nt.assert_equal((1, 4, 1),
                    get_bias_shape((True, False, True)))
    nt.assert_equal((3, 1, 5),
                    get_bias_shape((False, True, False)))


@nt.raises(AssertionError)
def test_add_bias_node_broadcastable_incorrect_size1():
    tn.SequentialNode("s", [
        tn.InputNode("in", shape=(3, 4, 5)),
        tn.AddBiasNode("b", broadcastable=(True, False))
    ]).network().build()


@nt.raises(AssertionError)
def test_add_bias_node_broadcastable_incorrect_size2():
    tn.SequentialNode("s", [
        tn.InputNode("in", shape=(3, 4, 5)),
        tn.AddBiasNode("b", broadcastable=(True, False, True, False))
    ]).network().build()


def test_add_bias_node():
    network = tn.SequentialNode("s", [
        tn.InputNode("in", shape=(3, 4, 5)),
        tn.AddBiasNode("b", broadcastable_axes=())
    ]).network()
    bias_var = network["b"].get_vw("bias")
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    y = np.random.randn(3, 4, 5).astype(fX)
    # test that bias is 0 initially
    np.testing.assert_allclose(fn(x)[0], x)
    # set bias_var value to new value
    bias_var.value = y
    # test that adding works
    np.testing.assert_allclose(fn(x)[0], x + y)


def test_linear_mapping_node_shape():
    def get_shapes(output_dim):
        network = tn.SequentialNode("s", [
            tn.InputNode("in", shape=(3, 4, 5)),
            tn.LinearMappingNode("linear", output_dim=output_dim),
        ]).network()
        weight_shape = network["linear"].get_vw("weight").shape
        output_shape = network["s"].get_vw("default").shape
        return weight_shape, output_shape

    nt.assert_equal(((5, 10), (3, 4, 10)), get_shapes(10))
    nt.assert_equal(((5, 1), (3, 4, 1)),
                    get_shapes(1))


def test_linear_mapping_node():
    network = tn.SequentialNode("s", [
        tn.InputNode("in", shape=(3, 4, 5)),
        tn.LinearMappingNode("linear", output_dim=6),
    ]).network()
    weight_var = network["linear"].get_vw("weight")
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    W = np.random.randn(5, 6).astype(fX)
    # test that weight is 0 initially
    np.testing.assert_allclose(fn(x)[0], np.zeros((3, 4, 6)))
    # set weight_var value to new value
    weight_var.value = W
    # test that adding works
    np.testing.assert_allclose(np.dot(x, W), fn(x)[0], rtol=1e-4, atol=1e-7)


def test_apply_node():
    network = tn.SequentialNode("s", [
        tn.InputNode("in", shape=(3, 4, 5)),
        tn.ApplyNode("a", fn=T.sum, shape_fn=lambda x: ()),
    ]).network()
    fn = network.function(["in"], ["s"])
    x = np.random.randn(3, 4, 5).astype(fX)
    np.testing.assert_allclose(fn(x)[0],
                               x.sum(),
                               rtol=1e-5)
