"""
relu where each channel has a different leak rate
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("interval_relu")
class IntervalReLUNode(treeano.NodeImpl):

    hyperparameter_names = ("leak_min",
                            "leak_max")

    def compute_output(self, network, in_vw):
        leak_min = network.find_hyperparameter(["leak_min"], 0)
        leak_max = network.find_hyperparameter(["leak_max"], 1)
        num_channels = in_vw.shape[1]
        alpha = np.linspace(leak_min, leak_max, num_channels).astype(fX)
        pattern = ["x" if i != 1 else 0 for i in range(in_vw.ndim)]
        alpha_var = T.constant(alpha).dimshuffle(*pattern)
        out_var = treeano.utils.rectify(in_vw.variable,
                                        negative_coefficient=alpha_var)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )
