"""
from "Visualizing and Understanding Convolutional Networks"
http://arxiv.org/abs/1311.2901
"""

import treeano
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
