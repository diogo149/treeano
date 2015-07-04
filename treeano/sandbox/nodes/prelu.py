"""
from "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
ImageNet Classification"
http://arxiv.org/abs/1502.01852
"""
import toolz
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


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
        inits = list(toolz.concat(network.find_hyperparameters(
            ["inits"],
            [treeano.inits.ConstantInit(initial_alpha)])))

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
        alpha_vw = network.create_variable(
            "alpha",
            is_shared=True,
            shape=shape,
            tags={"parameter", "bias"},
            inits=inits,
        )
        alpha = T.patternbroadcast(alpha_vw.variable, broadcastable)

        # return output
        network.create_variable(
            "default",
            variable=treeano.utils.rectify(in_vw.variable,
                                           negative_coefficent=alpha),
            shape=in_vw.shape,
            tags={"output"},
        )
