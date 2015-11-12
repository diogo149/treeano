import numpy as np
import theano
import theano.tensor as T

from .. import core
from .. import utils


@core.register_node("repeat_n_d")
class RepeatNDNode(core.NodeImpl):

    """
    repeats each axis of a tensor by a specified amount

    also referred to as "nearest neighbor" upsampling

    the normalize hyperparameter divides the output by the upsample factor,
    so that the total sum of the input and output are the same
    """

    hyperparameter_names = ("upsample_factor", "normalize")

    def compute_output(self, network, in_vw):
        upsample_factor = network.find_hyperparameter(["upsample_factor"])
        assert len(upsample_factor) == in_vw.ndim
        normalize = network.find_hyperparameter(["normalize"], False)

        out_var = in_vw.variable
        out_shape = list(in_vw.shape)
        # as of 20150922, performing the upsampling in reverse order
        # was faster
        for axis, factor in reversed(list(enumerate(upsample_factor))):
            # only handle this case for now
            assert utils.is_integral(factor)
            if factor != 1:
                out_var = T.extra_ops.repeat(out_var, factor, axis)
                if out_shape[axis] is not None:
                    out_shape[axis] *= factor

        if normalize:
            out_var = out_var / utils.as_fX(np.prod(upsample_factor))

        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )


def SpatialRepeatNDNode(name, upsample_factor, **kwargs):
    """
    same as RepeatNDNode, but assumes first 2 axes will not be upsampled
    """
    return RepeatNDNode(name,
                        upsample_factor=(1, 1) + upsample_factor,
                        **kwargs)


@core.register_node("sparse_upsample")
class SparseUpsampleNode(core.NodeImpl):

    """
    returns the tensor with 0s surrounding it in the specified dimensions

    also referred to as "perforated" upsampling
    """

    hyperparameter_names = ("upsample_factor",)

    def compute_output(self, network, in_vw):
        upsample_factor = network.find_hyperparameter(["upsample_factor"])
        assert len(upsample_factor) == in_vw.ndim

        out_symbolic_shape = list(in_vw.symbolic_shape())
        out_shape = list(in_vw.shape)
        slices = [slice(None) for _ in range(in_vw.ndim)]
        for axis, factor in reversed(list(enumerate(upsample_factor))):
            # only handle this case for now
            assert utils.is_integral(factor)
            if factor != 1:
                if out_shape[axis] is not None:
                    out_shape[axis] *= factor
                out_symbolic_shape[axis] *= factor
                slices[axis] = slice(None, None, factor)
        zeros = T.zeros(tuple(out_symbolic_shape))
        out_var = T.set_subtensor(zeros[tuple(slices)], in_vw.variable)

        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )


def SpatialSparseUpsampleNode(name, upsample_factor):
    """
    same as SparseUpsampleNode, but assumes first 2 axes will not be upsampled
    """
    return SparseUpsampleNode(name, upsample_factor=(1, 1) + upsample_factor)
