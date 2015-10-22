"""
from "Striving for Simplicity - The All Convolutional Net"
http://arxiv.org/abs/1412.6806

"""

import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils


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
