import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.nodes.auxiliary_costs as auxiliary_costs

fX = theano.config.floatX


def test_auxiliary_dense_softmax_cce_node_serialization():
    tn.check_serialization(
        auxiliary_costs.AuxiliaryDenseSoftmaxCCENode("a", {}))
    tn.check_serialization(
        auxiliary_costs.AuxiliaryDenseSoftmaxCCENode("a", {}, num_units=100))


def test_auxiliary_dense_softmax_cce_node():
    network = tn.SequentialNode(
        "seq",
        [tn.InputNode("in", shape=(3, 5)),
         auxiliary_costs.AuxiliaryDenseSoftmaxCCENode(
             "aux",
             {"target": tn.ConstantNode("target", value=np.eye(3).astype(fX))},
             num_units=3,
             cost_reference="foo"),
         tn.IdentityNode("i"),
         tn.InputElementwiseSumNode("foo", ignore_default_input=True)]
    ).network()
    x = np.random.randn(3, 5).astype(fX)
    fn = network.function(["in"], ["i", "foo", "aux_dense"])
    res = fn(x)
    np.testing.assert_equal(res[0], x)
    loss = T.nnet.categorical_crossentropy(
        np.ones((3, 3), dtype=fX) / 3.0,
        np.eye(3).astype(fX),
    ).mean().eval()
    np.testing.assert_allclose(res[1], loss)
