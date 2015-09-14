"""
nodes which behave similar to theano functions
"""

import theano
import theano.tensor as T

from .. import core
from .. import theano_extensions

fX = theano.config.floatX


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
        network.create_variable(
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

        network.create_variable(
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
        old_shape = in_vw.shape
        # TODO handle case when -1 is given
        assert -1 not in new_shape
        # TODO handle case when None is given
        assert None not in new_shape
        # FIXME
        # out_var = T.reshape(in_vw.variable,
        #                     newshape=new_shape,
        #                     ndim=len(new_shape))
        out_var = in_vw.variable.reshape(new_shape)
        network.create_variable(
            "default",
            variable=out_var,
            shape=new_shape,
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
        network.create_variable(
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
        network.create_variable(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )
