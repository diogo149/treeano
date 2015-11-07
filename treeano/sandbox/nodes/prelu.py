"""
from "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
ImageNet Classification"
http://arxiv.org/abs/1502.01852
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("prelu")
class PReLUNode(treeano.NodeImpl):

    hyperparameter_names = (
        "initial_alpha",
        # which axes should have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "parameter_axes",
        # which axes should not have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "non_parameter_axes",)

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        initial_alpha = network.find_hyperparameter(
            ["initial_alpha"],
            0.25)

        # calculate_shape
        ndim = in_vw.ndim
        parameter_axes = treeano.utils.find_axes(
            network,
            ndim,
            positive_keys=["parameter_axes"],
            negative_keys=["non_parameter_axes"],
            positive_default=[treeano.utils.nth_non_batch_axis(network, 0)])
        broadcastable = tuple([i not in parameter_axes
                               for i in range(ndim)])
        shape = tuple([1 if b else s
                       for b, s in zip(broadcastable, in_vw.shape)])

        # create state
        alpha_vw = network.create_vw(
            "alpha",
            is_shared=True,
            shape=shape,
            tags={"parameter", "bias"},
            default_inits=[treeano.inits.ConstantInit(initial_alpha)],
        )
        alpha = T.patternbroadcast(alpha_vw.variable, broadcastable)

        # return output
        network.create_vw(
            "default",
            variable=treeano.utils.rectify(in_vw.variable,
                                           negative_coefficient=alpha),
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("dropout_prelu")
class DropoutPReLUNode(treeano.NodeImpl):

    """
    NOTE: this was not part of the original PReLU paper, and is simply
    something I tried

    HACK most of this code was copied from PReLUNode
    """

    hyperparameter_names = (
        "initial_alpha",
        # which axes should have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "parameter_axes",
        # which axes should not have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "non_parameter_axes",
        # HACK this line is new:
        "dropout_prelu_probability",)

    # HACK this function is new
    def dropout_alpha(self, network, alpha_vw):
        alpha = alpha_vw.variable

        if network.find_hyperparameter(["deterministic"]):
            return alpha

        p = network.find_hyperparameter(["dropout_prelu_probability"], 0)
        if p == 0:
            return alpha

        keep_p = 1 - p
        from theano.sandbox.rng_mrg import MRG_RandomStreams
        srng = MRG_RandomStreams()
        # TODO should the kept values be amplified
        mask = srng.binomial(alpha_vw.shape, p=keep_p, dtype=fX)
        # TODO should the mask be broadcast as well
        return alpha * mask

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        initial_alpha = network.find_hyperparameter(
            ["initial_alpha"],
            0.25)

        # calculate_shape
        ndim = in_vw.ndim
        parameter_axes = treeano.utils.find_axes(
            network,
            ndim,
            positive_keys=["parameter_axes"],
            negative_keys=["non_parameter_axes"],
            positive_default=[treeano.utils.nth_non_batch_axis(network, 0)])
        broadcastable = tuple([i not in parameter_axes
                               for i in range(ndim)])
        shape = tuple([1 if b else s
                       for b, s in zip(broadcastable, in_vw.shape)])

        # create state
        alpha_vw = network.create_vw(
            "alpha",
            is_shared=True,
            shape=shape,
            tags={"parameter", "bias"},
            default_inits=[treeano.inits.ConstantInit(initial_alpha)],
        )
        # HACK this line is new:
        alpha = self.dropout_alpha(network, alpha_vw)
        alpha = T.patternbroadcast(alpha, broadcastable)

        # return output
        network.create_vw(
            "default",
            variable=treeano.utils.rectify(in_vw.variable,
                                           negative_coefficient=alpha),
            shape=in_vw.shape,
            tags={"output"},
        )
