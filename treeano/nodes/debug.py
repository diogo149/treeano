"""
nodes to help debugging
"""

import theano
import theano.tensor as T

from .. import utils
from .. import core


@core.register_node("print")
class PrintNode(core.NodeImpl):

    hyperparameter_names = ("message",)

    def compute_output(self, network, in_vw):
        message = network.find_hyperparameter(["message"], self.name)
        # TODO add attrs as hyperparameter for debugging
        out_var = theano.printing.Print(message)(in_vw.variable)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )
