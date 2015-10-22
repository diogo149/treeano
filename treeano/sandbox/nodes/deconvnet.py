"""
from "Visualizing and Understanding Convolutional Networks"
http://arxiv.org/abs/1311.2901
"""

import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils


class _Deconvnet(treeano.sandbox.utils.OverwriteGrad):

    """
    based on Lasagne Recipes on Guided Backpropagation
    """

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        if False:
            # explicitly rectify
            return (grd * (grd > 0).astype(inp.dtype),)
        else:
            # use the given fn
            return (self.fn(grd),)


deconvnet_relu = _Deconvnet(treeano.utils.rectify)


@treeano.register_node("deconvnet_relu")
class DeconvnetReLUNode(tn.BaseActivationNode):

    def activation(self, network, in_vw):
        return deconvnet_relu(in_vw.variable)
