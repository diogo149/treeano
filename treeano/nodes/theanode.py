"""
nodes which behave similar to theano functions
"""

import theano
import theano.tensor as T

from .. import core

fX = theano.config.floatX


@core.register_node("tile")
class TileNode(core.NodeImpl):

    """
    like theano.tensor.tile
    """

    hyperparameter_names = ("reps",)

    def compute_output(self, network, in_var):
        reps = network.find_hyperparameter(["reps"])
        shape = in_var.shape
        v = in_var.variable
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

    def compute_output(self, network, in_var):
        nb_class = network.find_hyperparameter(["nb_class"])
        cast_int32 = network.find_hyperparameter(["cast_int32"], False)
        dtype = network.find_hyperparameter(["dtype"], None)

        v = in_var.variable
        if cast_int32:
            v = v.astype("int32")

        network.create_variable(
            "default",
            variable=T.extra_ops.to_one_hot(v, nb_class=nb_class, dtype=dtype),
            shape=in_var.shape + (nb_class,),
            tags={"output"},
        )
