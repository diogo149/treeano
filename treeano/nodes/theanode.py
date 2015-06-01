"""
nodes which behave similar to theano functions
"""

import theano
import theano.tensor as T

from .. import core


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
