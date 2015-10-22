"""
from "Striving for Simplicity - The All Convolutional Net"
http://arxiv.org/abs/1412.6806

"""

import treeano
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
