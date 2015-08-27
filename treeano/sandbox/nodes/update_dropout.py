"""
technique that randomly 0's out the update deltas for each parameter
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("update_dropout")
class UpdateDropoutNode(treeano.Wrapper1NodeImpl):

    hyperparameter_names = ("update_dropout_probability",
                            "rescale_updates")

    def mutate_update_deltas(self, network, update_deltas):
        if network.find_hyperparameter(["deterministic"]):
            return
        p = network.find_hyperparameter(["update_dropout_probability"], 0)
        if p == 0:
            return
        rescale_updates = network.find_hyperparameter(["rescale_updates"],
                                                      False)
        keep_prob = 1 - p
        rescale_factor = 1 / (1 - p)
        srng = MRG_RandomStreams()
        # TODO parameterize search tags (to affect not only "parameters"s)
        vws = network.find_vws_in_subtree(tags={"parameter"},
                                          is_shared=True)
        for vw in vws:
            if vw.variable not in update_deltas:
                continue
            mask = srng.binomial(size=(), p=keep_prob, dtype=fX)
            if rescale_updates:
                mask *= rescale_factor
            update_deltas[vw.variable] *= mask
