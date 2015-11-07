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
        rescale_factor = 1 / keep_prob
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


@treeano.register_node("momentum_update_dropout")
class MomentumUpdateDropoutNode(treeano.Wrapper1NodeImpl):

    """
    randomly 0's out the update deltas for each parameter with momentum

    like update dropout, but with some probability (momentum), whether
    or not an update is dropped out is kept the same as the previous
    iteration
    """

    hyperparameter_names = ("update_dropout_probability",
                            "rescale_updates",
                            "update_dropout_momentum")

    def mutate_update_deltas(self, network, update_deltas):
        if network.find_hyperparameter(["deterministic"]):
            return
        p = network.find_hyperparameter(["update_dropout_probability"], 0)
        if p == 0:
            return
        rescale_updates = network.find_hyperparameter(["rescale_updates"],
                                                      False)
        momentum = network.find_hyperparameter(["update_dropout_momentum"])

        keep_prob = 1 - p
        rescale_factor = 1 / keep_prob
        srng = MRG_RandomStreams()
        # TODO parameterize search tags (to affect not only "parameters"s)
        vws = network.find_vws_in_subtree(tags={"parameter"},
                                          is_shared=True)
        for vw in vws:
            if vw.variable not in update_deltas:
                continue
            is_kept = network.create_vw(
                "momentum_update_dropout_is_kept(%s)" % vw.name,
                shape=(),
                is_shared=True,
                tags={"state"},
                # TODO: Should this be a random bool with prob p for each?
                default_inits=[treeano.inits.ConstantInit(1)]).variable

            keep_mask = srng.binomial(size=(), p=keep_prob, dtype=fX)
            momentum_mask = srng.binomial(size=(), p=momentum, dtype=fX)

            new_is_kept = (momentum_mask * is_kept
                           + (1 - momentum_mask) * keep_mask)

            mask = new_is_kept
            if rescale_updates:
                mask *= rescale_factor

            update_deltas[is_kept] = new_is_kept - is_kept
            update_deltas[vw.variable] *= mask
