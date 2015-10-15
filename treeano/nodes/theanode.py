"""
nodes which behave similar to theano functions
"""

import theano
import theano.tensor as T

from .. import core
from .. import theano_extensions

fX = theano.config.floatX


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
