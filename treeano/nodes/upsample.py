import theano
import theano.tensor as T

from .. import core
from .. import utils


@core.register_node("repeat_n_d")
class RepeatNDNode(core.NodeImpl):

    """
    repeats each axis of a tensor by a specified amount
    """

    hyperparameter_names = ("upsample_factor",)

    def compute_output(self, network, in_vw):
        upsample_factor = network.find_hyperparameter(["upsample_factor"])
        assert len(upsample_factor) == in_vw.ndim

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

        network.create_variable(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )


def SpatialRepeatNDNode(name, upsample_factor):
    """
    same as RepeatNDNode, but assumes first 2 axes will not be upsampled
    """
    return RepeatNDNode(name, upsample_factor=(1, 1) + upsample_factor)
