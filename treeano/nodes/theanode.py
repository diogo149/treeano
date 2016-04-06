"""
nodes which behave similar to theano functions
"""

import numpy as np
import theano
import theano.tensor as T

from .. import utils
from .. import core
from .. import theano_extensions

fX = theano.config.floatX


@core.register_node("clip")
class ClipNode(core.NodeImpl):

    """
    like theano.tensor.clip
    """

    hyperparameter_names = ("bounds",)

    def compute_output(self, network, in_vw):
        bounds = network.find_hyperparameter(["bounds"])
        network.create_vw(
            "default",
            variable=T.clip(in_vw.variable, *bounds),
            shape=in_vw.shape,
            tags={"output"},
        )


@core.register_node("swapaxes")
class SwapAxesNode(core.NodeImpl):

    """
    like theano.tensor.swapaxes
    """

    hyperparameter_names = ("axes",)

    def compute_output(self, network, in_vw):
        axis1, axis2 = network.find_hyperparameter(["axes"])
        out_shape = list(in_vw.shape)
        out_shape[axis1], out_shape[axis2] = out_shape[axis2], out_shape[axis1]
        network.create_vw(
            "default",
            variable=T.swapaxes(in_vw.variable, axis1, axis2),
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("sqr")
class SqrNode(core.NodeImpl):

    """
    like theano.tensor.sqr
    """

    def compute_output(self, network, in_vw):
        network.create_vw(
            "default",
            variable=T.sqr(in_vw.variable),
            shape=in_vw.shape,
            tags={"output"},
        )


@core.register_node("sqrt")
class SqrtNode(core.NodeImpl):

    """
    like theano.tensor.sqrt
    """

    def compute_output(self, network, in_vw):
        network.create_vw(
            "default",
            variable=T.sqrt(in_vw.variable),
            shape=in_vw.shape,
            tags={"output"},
        )


@core.register_node("tile")
class TileNode(core.NodeImpl):

    """
    like theano.tensor.tile
    """

    hyperparameter_names = ("reps",)

    def compute_output(self, network, in_vw):
        reps = network.find_hyperparameter(["reps"])
        shape = in_vw.shape
        v = in_vw.variable
        network.create_vw(
            "default",
            variable=T.tile(v, reps),
            shape=tuple(s * r for s, r in zip(shape, reps)),
            tags={"output"},
        )


@core.register_node("to_one_hot")
class ToOneHotNode(core.NodeImpl):

    """
    like theano.tensor.extra_ops.to_one_hot
    """

    hyperparameter_names = ("nb_class",
                            "cast_int32",
                            "dtype")

    def compute_output(self, network, in_vw):
        nb_class = network.find_hyperparameter(["nb_class"])
        cast_int32 = network.find_hyperparameter(["cast_int32"], False)
        dtype = network.find_hyperparameter(["dtype"], None)

        v = in_vw.variable
        if cast_int32:
            v = v.astype("int32")

        network.create_vw(
            "default",
            variable=T.extra_ops.to_one_hot(v, nb_class=nb_class, dtype=dtype),
            shape=in_vw.shape + (nb_class,),
            tags={"output"},
        )


@core.register_node("reshape")
class ReshapeNode(core.NodeImpl):

    """
    like theano.tensor.reshape
    """

    hyperparameter_names = ("newshape",
                            "shape")

    def compute_output(self, network, in_vw):
        new_shape = network.find_hyperparameter(["newshape",
                                                 "shape"])
        # TODO use this to determine shape
        old_shape = in_vw.shape

        out_shape = new_shape

        # case if -1 is in new shape
        if -1 in new_shape:
            # should have only 1 -1
            assert sum([s == -1 for s in out_shape]) == 1
            out_shape = tuple([None if s == -1 else s for s in out_shape])

        # FIXME
        # out_var = T.reshape(in_vw.variable,
        #                     newshape=new_shape,
        #                     ndim=len(new_shape))
        out_var = in_vw.variable.reshape(new_shape)
        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("repeat")
class RepeatNode(core.NodeImpl):

    """
    like theano.tensor.repeat
    """

    hyperparameter_names = ("repeats",
                            "axis")

    def compute_output(self, network, in_vw):
        repeats = network.find_hyperparameter(["repeats"])
        axis = network.find_hyperparameter(["axis"])

        out_shape = list(in_vw.shape)
        if out_shape[axis] is not None:
            out_shape[axis] *= repeats
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=T.repeat(in_vw.variable, repeats=repeats, axis=axis),
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("dimshuffle")
class DimshuffleNode(core.NodeImpl):

    """
    like dimshuffle
    """

    hyperparameter_names = ("pattern",)

    def compute_output(self, network, in_vw):
        pattern = network.find_hyperparameter(["pattern"])
        out_var = in_vw.variable.dimshuffle(*pattern)
        out_shape = tuple([1 if i == "x" else in_vw.shape[i]
                           for i in pattern])
        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("gradient_reversal")
class GradientReversalNode(core.NodeImpl):

    """
    like treeano.theano_extensions.gradient.gradient_reversal
    """

    def compute_output(self, network, in_vw):
        out_var = theano_extensions.gradient.gradient_reversal(in_vw.variable)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("zero_grad")
class ZeroGradNode(core.NodeImpl):

    """
    like theano.gradient.zero_grad
    """

    def compute_output(self, network, in_vw):
        out_var = theano.gradient.zero_grad(in_vw.variable)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("disconnected_grad")
class DisconnectedGradNode(core.NodeImpl):

    """
    like theano.gradient.disconnected_grad
    """

    def compute_output(self, network, in_vw):
        out_var = theano.gradient.disconnected_grad(in_vw.variable)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("mean")
class MeanNode(core.NodeImpl):

    """
    like theano.tensor.mean
    """

    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], None)
        out_var = T.mean(in_vw.variable, axis=axis)

        if axis is None:
            out_shape = ()
        elif isinstance(axis, int):
            out_shape = list(in_vw.shape)
            out_shape.pop(axis)
            out_shape = tuple(out_shape)
        else:
            raise NotImplementedError()

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("max")
class MaxNode(core.NodeImpl):

    """
    like theano.tensor.max
    """

    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], None)
        out_var = T.max(in_vw.variable, axis=axis)

        if axis is None:
            out_shape = ()
        elif isinstance(axis, int):
            out_shape = list(in_vw.shape)
            out_shape.pop(axis)
            out_shape = tuple(out_shape)
        else:
            raise NotImplementedError()

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("sum")
class SumNode(core.NodeImpl):

    """
    like theano.tensor.sum
    """

    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], None)
        out_var = T.sum(in_vw.variable, axis=axis)

        if axis is None:
            out_shape = ()
        elif isinstance(axis, int):
            out_shape = list(in_vw.shape)
            out_shape.pop(axis)
            out_shape = tuple(out_shape)
        else:
            raise NotImplementedError()

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("flatten")
class FlattenNode(core.NodeImpl):

    """
    like theano.tensor.flatten
    """

    hyperparameter_names = ("outdim",)

    def compute_output(self, network, in_vw):
        outdim = network.find_hyperparameter(["outdim"])

        out_var = T.flatten(in_vw.variable, outdim=outdim)

        trailing_axes = in_vw.shape[outdim - 1:]
        if any(a is None for a in trailing_axes):
            final_size = None
        else:
            final_size = np.prod(trailing_axes)

        out_shape = in_vw.shape[:outdim - 1] + (final_size,)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("add_broadcast")
class AddBroadcastNode(core.NodeImpl):

    """
    like theano.tensor.addbroadcast
    """

    hyperparameter_names = ("axes",)

    def compute_output(self, network, in_vw):
        axes = network.find_hyperparameter(["axes"])
        out_var = T.addbroadcast(in_vw.variable, *axes)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("pow")
class PowNode(core.NodeImpl):

    """
    like theano.tensor.pow
    """

    hyperparameter_names = ("exponent",)

    def compute_output(self, network, in_vw):
        exponent = network.find_hyperparameter(["exponent"])
        network.create_vw(
            "default",
            variable=T.pow(in_vw.variable, exponent),
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("pad")
class PadNode(core.NodeImpl):

    """
    like treeano.theano_extensions.padding.pad
    """

    hyperparameter_names = ("padding",)

    def compute_output(self, network, in_vw):
        padding = network.find_hyperparameter(["padding"])

        out_shape = list(in_vw.shape)
        for i in range(in_vw.ndim):
            if out_shape[i] is not None:
                # FIXME make work for asymmetric padding
                out_shape[i] += 2 * padding[i]
        network.create_vw(
            "default",
            variable=theano_extensions.padding.pad(in_vw.variable, padding),
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("cumsum")
class CumsumNode(core.NodeImpl):

    """
    like theano.tensor.cumsum
    """

    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        network.create_vw(
            "default",
            variable=T.cumsum(in_vw.variable, axis=axis),
            shape=in_vw.shape,
            tags={"output"}
        )


@core.register_node("index")
class IndexNode(core.NodeImpl):

    """
    like indexing a theano tensor
    """

    hyperparameter_names = ("idxs",)

    def compute_output(self, network, in_vw):
        idxs = network.find_hyperparameter(["idxs"])

        out_shape = []
        for i, s in enumerate(in_vw.shape):
            if i >= len(idxs):
                # copy original shape if idxs does not have same length
                # as shape
                out_shape.append(s)
                continue
            else:
                idx = idxs[i]

            if isinstance(idx, slice):
                if s is None:
                    # we don't know input or output size
                    out_shape.append(None)
                else:
                    # calculate number of indices
                    out_shape.append(len(np.zeros(s)[idx]))
            elif utils.is_integral(idx):
                # lose this axis
                pass
            else:
                # TODO can handle cases when idx is a tensor
                raise ValueError

        out_var = in_vw.variable[tuple(idxs)]
        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )
