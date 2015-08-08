import numpy as np
import theano
import theano.tensor as T

from .. import utils


class PercentileOp(theano.Op):

    """
    like numpy.percentile
    returns q-th percentile of the data for q in [0, 100]
    """

    # TODO can implement gradient w.r.t. q

    __props__ = ("axis", "keepdims")

    def __init__(self, axis, keepdims):
        if isinstance(axis, list):
            axis = tuple(axis)
        assert axis is None or isinstance(axis, (int, tuple))
        self.axis = axis
        self.keepdims = keepdims

    def make_node(self, a, q):
        # cast q to theano variable
        if isinstance(q, (int, float)):
            scalar_type = T.scalar().type
            q = T.Constant(scalar_type, q)

        # set to all axes if none specified
        if self.axis is None:
            axis = range(a.ndim)
        elif isinstance(self.axis, int):
            axis = [self.axis]
        else:
            axis = self.axis

        # calculate broadcastable
        if self.keepdims:
            broadcastable = [b or (ax in axis)
                             for ax, b in enumerate(a.broadcastable)]
        else:
            broadcastable = [b
                             for ax, b in enumerate(a.broadcastable)
                             if ax not in axis]

        out = T.TensorType(a.dtype, broadcastable)()
        return theano.gof.Apply(self, [a, q], [out])

    def perform(self, node, inputs, output_storage):
        a, q = inputs
        z, = output_storage
        res = np.percentile(a, q, axis=self.axis, keepdims=self.keepdims)
        z[0] = utils.as_fX(res)


def percentile(a, q, axis=None, keepdims=False):
    return PercentileOp(axis, keepdims)(a, q)
