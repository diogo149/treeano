from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T

from .. import utils
from .. import core


@core.register_node("embedding")
class EmbeddingNode(core.NodeImpl):

    hyperparameter_names = ("input_size",
                            "output_size")

    def compute_output(self, network, in_vw):
        input_size = network.find_hyperparameter(["input_size"])
        output_size = network.find_hyperparameter(["output_size"])
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=(input_size, output_size),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable

        out_shape = in_vw.shape + (output_size,)
        out_ss = in_vw.symbolic_shape() + (output_size,)

        assert in_vw.dtype == "int32"
        out_var = W[in_vw.variable.ravel()]
        out_var = out_var.reshape(out_ss)

        network.create_vw(
            name="default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
