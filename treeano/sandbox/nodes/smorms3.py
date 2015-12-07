"""
SMORMS3 algorithm (squared mean over root mean squared cubed)
based on http://sifter.org/~simon/journal/20150420.html
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("smorms3")
class SMORMS3Node(tn.StandardUpdatesNode):

    hyperparameter_names = ("learning_rate",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 0.001)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-16)
        update_deltas = treeano.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            mem_vw = network.create_vw(
                "smorms3_mem(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[treeano.inits.ConstantInit(1)],
            )
            g_vw = network.create_vw(
                "smorms3_g(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )
            g2_vw = network.create_vw(
                "smorms3_g2(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )
            parameter = parameter_vw.variable
            mem = mem_vw.variable
            g = g_vw.variable
            g2 = g2_vw.variable
            r = 1 / (mem + 1)
            new_g = (1 - r) * g + r * grad
            new_g2 = (1 - r) * g2 + r * grad ** 2
            term1 = (new_g ** 2) / (new_g2 + epsilon)
            term2 = T.sqrt(new_g2) + epsilon
            parameter_delta = -grad * T.minimum(learning_rate, term1) / term2
            new_mem = 1 + mem * (1 - term1)
            update_deltas[parameter] = parameter_delta
            update_deltas[mem] = new_mem - mem
            update_deltas[g] = new_g - g
            update_deltas[g2] = new_g2 - g2
        return update_deltas
