import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import dNDF


fX = theano.config.floatX


def test_split_probabilities_to_leaf_probabilities_node_serialization():
    for node in [dNDF.TheanoSplitProbabilitiesToLeafProbabilitiesNode,
                 dNDF.NumpySplitProbabilitiesToLeafProbabilitiesNode]:
        tn.check_serialization(node("a"))


def test_split_probabilities_to_leaf_probabilities_node():
    x = np.array([[[0.9, 0.2],
                   [0.7, 0.6],
                   [0.4, 0.3]]],
                 dtype=fX)
    ans = np.array([[[0.9 * 0.7, 0.2 * 0.6],
                     [0.9 * (1 - 0.7), 0.2 * (1 - 0.6)],
                     [(1 - 0.9) * 0.4, (1 - 0.2) * 0.3],
                     [(1 - 0.9) * (1 - 0.4), (1 - 0.2) * (1 - 0.3)]]],
                   dtype=fX)

    for node in [dNDF.TheanoSplitProbabilitiesToLeafProbabilitiesNode,
                 dNDF.NumpySplitProbabilitiesToLeafProbabilitiesNode]:
        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(1, 3, 2)),
             node("p")]
        ).network()

        fn = network.function(["i"], ["s"])

        np.testing.assert_allclose(ans,
                                   fn(x)[0],
                                   rtol=1e-5)


def test_split_probabilities_to_leaf_probabilities_node_grad():
    x = np.array([[[0.9, 0.2],
                   [0.7, 0.6],
                   [0.4, 0.3]]],
                 dtype=fX)

    def node_to_grad(node):
        network = tn.SequentialNode(
            "s",
            [tn.InputNode("i", shape=(1, 3, 2)),
             node("p")]
        ).network()

        in_ = network["i"].get_vw("default").variable
        out = network["s"].get_vw("default").variable
        g = T.grad(out[:, 0].sum(), in_)
        fn = network.function(["i"], [g])
        return fn(x)[0]

    n1, n2 = [dNDF.TheanoSplitProbabilitiesToLeafProbabilitiesNode,
              dNDF.NumpySplitProbabilitiesToLeafProbabilitiesNode]

    np.testing.assert_allclose(node_to_grad(n1),
                               node_to_grad(n2),
                               rtol=1e-5)
