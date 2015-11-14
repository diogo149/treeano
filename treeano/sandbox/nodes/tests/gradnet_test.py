from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import gradnet

fX = theano.config.floatX


def test_grad_net_interpolation_node():
    network = tn.SequentialNode(
        "s",
        [tn.InputNode("i", shape=(1, 10)),
         gradnet.GradNetInterpolationNode(
             "gradnet",
             {"early": tn.ReLUNode("r"),
              "late": tn.TanhNode("t")},
             late_gate=0.5)]
    ).network()

    fn = network.function(["i"], ["s"])
    x = np.random.randn(1, 10).astype(fX)
    ans = 0.5 * np.clip(x, 0, np.inf) + 0.5 * np.tanh(x)
    np.testing.assert_allclose(ans, fn(x)[0], rtol=1e-5)


def test_grad_net_optimizer_interpolation_node():

    class StateNode(treeano.NodeImpl):
        input_keys = ()

        def compute_output(self, network):
            network.create_vw(
                name="default",
                shape=(),
                is_shared=True,
                tags=["parameter"],
                inits=[],
            )

    def updater(const):
        class UpdaterNode(treeano.nodes.updates.StandardUpdatesNode):

            def _new_update_deltas(self, network, vws, grads):
                return treeano.UpdateDeltas({vw.variable: const for vw in vws})

        return UpdaterNode

    network = tn.SharedHyperparameterNode(
        "n",
        gradnet.GradNetOptimizerInterpolationNode(
            "g",
            {"subtree": StateNode("s"),
             "cost": tn.ReferenceNode("r", reference="s")},
            early=updater(-1),
            late=updater(1)),
        hyperparameter="late_gate"
    ).network()

    fn1 = network.function([("n", "hyperparameter")],
                           [],
                           include_updates=True)
    fn2 = network.function([], ["n"])
    gates_and_answers = [(0, -1),
                         (0.25, -1.5),
                         (1, -0.5),
                         (1, 0.5)]
    for gate, ans in gates_and_answers:
        fn1(gate)
        np.testing.assert_allclose(ans, fn2()[0], rtol=1e-1)
