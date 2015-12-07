"""
based on https://en.wikipedia.org/wiki/Quickprop

standard quickprop has some issues:
- numerical stability when dividing by (prev_grad - grad)
- multiplying by the previous update
  - can be an issue if a previous update is 0

possible (partial) solutions:
- use momentum on the update
- add noise to the gradient (similar to "Adding Gradient Noise Improves
  Learning for Very Deep Networks" http://arxiv.org/abs/1511.06807)
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import treeano
import treeano.nodes as tn
from treeano.sandbox import update_utils

fX = theano.config.floatX


@treeano.register_node("quickprop")
class QuickpropNode(tn.StandardUpdatesNode):

    def _new_update_deltas(self, network, parameter_vws, grads):
        update_deltas = treeano.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            prev_grad, _ = update_utils.update_previous(
                network,
                update_deltas,
                grad,
                "grad(%s)" % parameter_vw.name,
                parameter_vw.shape)

            prev_update = network.create_vw(
                "quickprop_prev_update(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[treeano.inits.ConstantInit(1)],
            ).variable

            denom = prev_grad - grad
            # TODO paramerize
            epsilon = 1e-6
            denom = denom + treeano.utils.sign_non_zero(denom) * epsilon
            parameter_delta = prev_update * grad / denom

            parameter = parameter_vw.variable
            update_deltas[parameter] = parameter_delta
            update_deltas[prev_update] = parameter_delta - prev_update
        return update_deltas
