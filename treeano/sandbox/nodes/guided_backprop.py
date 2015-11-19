"""
from "Striving for Simplicity - The All Convolutional Net"
http://arxiv.org/abs/1412.6806
"""

import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils
import canopy


class _GuidedBackprop(treeano.sandbox.utils.OverwriteGrad):

    """
    based on Lasagne Recipes on Guided Backpropagation
    """

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)


guided_backprop_relu = _GuidedBackprop(treeano.utils.rectify)


@treeano.register_node("guided_backprop_relu")
class GuidedBackpropReLUNode(tn.BaseActivationNode):

    def activation(self, network, in_vw):
        return guided_backprop_relu(in_vw.variable)


def replace_relu_with_guided_backprop_transform(network,
                                                nodes=(tn.ReLUNode,),
                                                **kwargs):

    def inner(node):
        if isinstance(node, nodes):
            return GuidedBackpropReLUNode(node.name)
        else:
            return node

    return canopy.transforms.fns.transform_root_node_postwalk(
        network, inner, **kwargs)


class ReplaceReLUWithGuidedBackprop(canopy.handlers.NetworkHandlerImpl):

    def __init__(self, nodes=(tn.ReLUNode,)):
        self.nodes = nodes

    def transform_network(self, network):
        return replace_relu_with_guided_backprop_transform(network, self.nodes)

replace_relu_with_guided_backprop_handler = ReplaceReLUWithGuidedBackprop
