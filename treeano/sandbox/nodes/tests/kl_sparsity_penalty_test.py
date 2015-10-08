import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import kl_sparsity_penalty as sp

fX = theano.config.floatX


def test_elementwise_kl_sparsity_penalty_node_serialization():
    tn.check_serialization(sp.ElementwiseKLSparsityPenaltyNode("a"))


def test_auxiliary_kl_sparsity_penalty_node_serialization():
    tn.check_serialization(sp.AuxiliaryKLSparsityPenaltyNode("a"))


def test_elementwise_kl_sparsity_penalty_node1():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(5, 3)),
         sp.ElementwiseKLSparsityPenaltyNode("sp", sparsity=0.1)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.ones((5, 3), dtype=fX) * 0.1
    np.testing.assert_allclose(np.zeros((5, 3), dtype=fX),
                               fn(x)[0],
                               rtol=1e-5,
                               atol=1e-7)


def test_elementwise_kl_sparsity_penalty_node2():
    # just testing that it runs
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(10, 3)),
         tn.DenseNode("d", num_units=9),
         sp.ElementwiseKLSparsityPenaltyNode("sp", sparsity=0.1)]
    ).network()
    fn = network.function(["i"], ["s"])
    x = np.random.rand(10, 3).astype(fX)
    nt.assert_equal(fn(x)[0].shape, (10, 9))


def test_auxiliary_kl_sparsity_penalty_node():
    # testing that both sparsity penalty versions return the same thing
    network = tn.HyperparameterNode(
        "hp",
        tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(10, 3)),
             tn.DenseNode("d", num_units=9),
             sp.AuxiliaryKLSparsityPenaltyNode("scp", cost_reference="sum"),
             sp.ElementwiseKLSparsityPenaltyNode("sp"),
             tn.AggregatorNode("a"),
             # zero out rest of network, so that value of sum is just the value
             # from auxiliary sparsity pentalty node
             tn.ConstantNode("foo", value=0),
             tn.InputElementwiseSumNode("sum")]),
        sparsity=0.1,
    ).network()
    fn = network.function(["i"], ["sum", "a"])
    x = np.random.rand(10, 3).astype(fX)
    res = fn(x)
    np.testing.assert_equal(res[0], res[1])
