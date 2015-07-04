import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import contraction_penalty as cp

fX = theano.config.floatX


def test_elementwise_contraction_penalty_node_serialization():
    tn.check_serialization(cp.ElementwiseContractionPenaltyNode("a"))


def test_auxiliary_contraction_penalty_node_serialization():
    tn.check_serialization(cp.AuxiliaryContractionPenaltyNode(
        "a", tn.IdentityNode("b")))


def test_elementwise_contraction_penalty_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 3)),
         cp.ElementwiseContractionPenaltyNode("cp", input_reference="i")]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.rand(10, 3).astype(fX)
    # jacobian of each location is 1
    # squared jacobian is 1
    # mean squared jacobian is 1/3
    np.testing.assert_equal(fn(x)[0], np.ones(10, dtype=fX) / 3)


def test_elementwise_contraction_penalty_node2():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 3)),
         tn.DenseNode("d", num_units=9),
         cp.ElementwiseContractionPenaltyNode("cp", input_reference="i")]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.rand(10, 3).astype(fX)
    nt.assert_equal(fn(x)[0].shape, (10,))


def test_auxiliary_contraction_penalty_node():
    # testing that both contraction penalty versions return the same thing
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 3)),
         cp.AuxiliaryContractionPenaltyNode(
             "acp",
             tn.DenseNode("d", num_units=9),
             cost_reference="sum"),
         cp.ElementwiseContractionPenaltyNode("cp", input_reference="i"),
         tn.AggregatorNode("a"),
         # zero out rest of network, so that value of sum is just value from
         # auxiliary contraction pentalty node
         tn.ConstantNode("foo", value=0),
         tn.InputElementwiseSumNode("sum")]
    ).network()
    fn = network.function(["i"], ["sum", "a"])
    x = np.random.rand(10, 3).astype(fX)
    res = fn(x)
    np.testing.assert_equal(res[0], res[1])
