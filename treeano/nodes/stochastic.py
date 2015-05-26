import warnings

import theano
import theano.tensor as T

from .. import core
from .. import utils

floatX = theano.config.floatX


class DropoutNode(core.NodeImpl):

    """
    node that drops out random units
    """

    hyperparameter_names = ("dropout_probability",
                            "probability",
                            "p")

    def compute_output(self, network, in_vw):
        p = network.find_hyperparameter(["dropout_probability",
                                         "probability",
                                         "p"],
                                        0)
        if p == 0:
            network.copy_variable(
                name="default",
                previous_variable=in_vw,
                tags={"output"},
            )
        else:
            rescale_factor = 1 / (1 - p)
            mask_shape = in_vw.shape
            if any(s is None for s in mask_shape):
                # NOTE: this uses symbolic shape - can be an issue with
                # theano.clone and random numbers
                # https://groups.google.com/forum/#!topic/theano-users/P7Mv7Fg0kUs
                warnings.warn("using symbolic shape for dropout mask, "
                              "which can be an issue with theano.clone")
                mask_shape = in_vw.variable.shape
            mask = rescale_factor * utils.srng.binomial(mask_shape,
                                                        p=p,
                                                        dtype=floatX)
            network.create_variable(
                "default",
                variable=in_vw.variable * mask,
                shape=in_vw.shape,
                tags={"output"},
            )
