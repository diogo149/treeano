import numpy as np
import theano

from treeano import nodes

floatX = theano.config.floatX


def test_update_scale_node_serialization():
    # NOTE: setting shape to be list because of json serialization
    # (serializing a tuple results in a list)
    nodes.check_serialization(nodes.UpdateScaleNode(
        "a", nodes.InputNode("a", shape=[3, 4, 5])))


def test_update_scale_node():

    # testing constant updater
    network = nodes.toy.ConstantUpdaterNode(
        "cun",
        nodes.SequentialNode("seq", [
            nodes.InputNode("i", shape=(1, 2, 3)),
            nodes.DenseNode("fc", num_units=5)
        ]),
        value=5,
    ).network()
    ud = network.update_deltas
    assert ud[network["fc_linear"].get_variable("weight").variable] == 5

    # test update scale node
    network = nodes.toy.ConstantUpdaterNode(
        "cun",
        nodes.SequentialNode("seq", [
            nodes.InputNode("i", shape=(1, 2, 3)),
            nodes.UpdateScaleNode(
                "usn",
                nodes.DenseNode("fc", num_units=5),
                scale_factor=-2)
        ]),
        value=5,
    ).network()
    ud = network.update_deltas
    assert ud[network["fc_linear"].get_variable("weight").variable] == -10


def test_sgd_node():
    nodes.test_utils.check_updates_node(nodes.SGDNode, learning_rate=0.01)


def test_adam_node():
    nodes.test_utils.check_updates_node(nodes.AdamNode)
