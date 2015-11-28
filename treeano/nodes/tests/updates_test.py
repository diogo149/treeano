import numpy as np
import theano
import treeano
import treeano.nodes as tn

floatX = theano.config.floatX


def test_update_scale_node_serialization():
    # NOTE: setting shape to be list because of json serialization
    # (serializing a tuple results in a list)
    tn.check_serialization(tn.UpdateScaleNode(
        "a", tn.InputNode("a", shape=[3, 4, 5])))


def test_weight_decay_node_serialization():
    tn.check_serialization(tn.WeightDecayNode("a", tn.IdentityNode("b")))


def test_update_scale_node():

    # testing constant updater
    network = tn.toy.ConstantUpdaterNode(
        "cun",
        tn.SequentialNode("seq", [
            tn.InputNode("i", shape=(1, 2, 3)),
            tn.DenseNode("fc", num_units=5)
        ]),
        value=5,
    ).network()
    ud = network.update_deltas
    assert ud[network["fc_linear"].get_vw("weight").variable] == 5

    # test update scale node
    network = tn.toy.ConstantUpdaterNode(
        "cun",
        tn.SequentialNode("seq", [
            tn.InputNode("i", shape=(1, 2, 3)),
            tn.UpdateScaleNode(
                "usn",
                tn.DenseNode("fc", num_units=5),
                scale_factor=-2)
        ]),
        value=5,
    ).network()
    ud = network.update_deltas
    assert ud[network["fc_linear"].get_vw("weight").variable] == -10


def test_sgd_node():
    tn.test_utils.check_updates_node(tn.SGDNode, learning_rate=0.01)


def test_adam_node():
    tn.test_utils.check_updates_node(tn.AdamNode)


def test_adamax_node():
    tn.test_utils.check_updates_node(tn.AdaMaxNode)


def test_nesterovs_accelerated_gradient_node():
    tn.test_utils.check_updates_node(tn.NesterovsAcceleratedGradientNode,
                                     # NOTE: default learning rate can
                                     # cause divergence
                                     learning_rate=0.01)


def test_adadelta_node():
    tn.test_utils.check_updates_node(tn.ADADELTANode)


def test_adagrad_node():
    tn.test_utils.check_updates_node(tn.ADAGRADNode)


def test_rmsprop_node():
    tn.test_utils.check_updates_node(tn.RMSPropNode)


def test_rprop_node():
    tn.test_utils.check_updates_node(tn.RpropNode,
                                     learning_rate=0.001,
                                     rprop_type="Rprop")
    tn.test_utils.check_updates_node(tn.RpropNode,
                                     learning_rate=0.001,
                                     rprop_type="iRprop-")


def test_weight_decay_node():
    class WeightNode(treeano.NodeImpl):
        input_keys = ()

        def compute_output(self, network):
            network.create_vw(
                "default",
                is_shared=True,
                shape=(),
                tags={"weight", "parameter"},
                inits=[treeano.inits.ConstantInit(100.0)],
            )

    network = tn.WeightDecayNode(
        "a",
        WeightNode("b"),
        weight_decay=0.5
    ).network()
    fn1 = network.function([], ["a"])
    fn2 = network.function([], ["a"], include_updates=True)

    np.testing.assert_equal(fn1(), [100.0])
    np.testing.assert_equal(fn1(), [100.0])
    np.testing.assert_equal(fn2(), [100.0])
    np.testing.assert_equal(fn2(), [50.0])
    np.testing.assert_equal(fn2(), [25.0])
