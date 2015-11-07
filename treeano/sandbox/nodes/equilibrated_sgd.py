"""
from
"RMSProp and equilibrated adaptive learning rates for non-convex optimization"
http://arxiv.org/abs/1502.04390

NOTE: Rop doesn't work for many operations, and it often causes nan's
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("equilibrated_sgd")
class EquilibratedSGDNode(tn.StandardUpdatesNode):

    hyperparameter_names = ("learning_rate",
                            "damping_factor")

    def _new_update_deltas(self, network, parameter_vws, grads):
        # NOTE: in the paper, learning_rate is referred to as epsilon
        # not doing that here as it would be confusing
        learning_rate = network.find_hyperparameter(["learning_rate"], 0.01)
        # NOTE: this is referred to as lambda in the paper
        # NOTE: when doing hyperparameter selection in the paper,
        # they select from 1e-4, 1e-5, 1e-6
        damping_factor = network.find_hyperparameter(["damping_factor"], 1e-2)

        update_deltas = treeano.UpdateDeltas()

        k_vw = network.create_vw(
            "esgd_count",
            shape=(),
            is_shared=True,
            tags={"state"},
            default_inits=[],
        )
        k = k_vw.variable
        new_k = k + 1
        update_deltas[k] = new_k - k

        for parameter_vw, grad in zip(parameter_vws, grads):
            D_vw = network.create_vw(
                "esgd_D(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )

            # TODO ESGD update should only occur every 20 iterations
            # to amortize cost
            parameter = parameter_vw.variable
            D = D_vw.variable
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            # noise vector
            v = srng.normal(size=parameter.shape)
            Hv = T.Rop(grad, parameter, v)
            D_delta = T.sqr(Hv)
            new_D = D + D_delta
            # new_D / new_k is essentially a mean
            denominator = damping_factor + T.sqrt(new_D / new_k)
            parameter_delta = -learning_rate * grad / denominator
            update_deltas[parameter] = parameter_delta
            update_deltas[D] = D_delta
        return update_deltas
