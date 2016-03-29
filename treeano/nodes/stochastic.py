import warnings

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from .. import core
from .. import utils

fX = theano.config.floatX


# TODO: Refactor to extract shared logic from these nodes.
@core.register_node("dropout")
class DropoutNode(core.NodeImpl):

    """
    node that drops out random units
    """

    hyperparameter_names = ("dropout_probability",
                            "probability",
                            "p",
                            "deterministic")

    def compute_output(self, network, in_vw):
        deterministic = network.find_hyperparameter(["deterministic"])
        p = network.find_hyperparameter(["dropout_probability",
                                         "probability",
                                         "p"],
                                        0)
        if deterministic or p == 0:
            network.copy_vw(
                name="default",
                previous_vw=in_vw,
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
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            # set bernoulli probability to be inverse of dropout probability
            # because 1 means to keep the unit
            bernoulli_prob = 1 - p
            mask = rescale_factor * srng.binomial(mask_shape,
                                                  p=bernoulli_prob,
                                                  dtype=fX)
            network.create_vw(
                "default",
                variable=in_vw.variable * mask,
                shape=in_vw.shape,
                tags={"output"},
            )


@core.register_node("gaussian_dropout")
class GaussianDropoutNode(core.NodeImpl):

    """
    node that adds gaussian noise to units
    """

    hyperparameter_names = ("sigma",
                            "dropout_probability",
                            "probability",
                            "p",
                            "deterministic")

    def compute_output(self, network, in_vw):
        deterministic = network.find_hyperparameter(["deterministic"])
        sigma = network.find_hyperparameter(["sigma"], None)
        if sigma is None:
            p = network.find_hyperparameter(["dropout_probability",
                                             "probability",
                                             "p"],
                                            0)
            if p == 0:
                sigma = 0
            else:
                # derive gaussian dropout variance from bernoulli dropout
                # probability
                sigma = T.sqrt(p / (1 - p))
        if deterministic or sigma == 0:
            network.copy_vw(
                name="default",
                previous_vw=in_vw,
                tags={"output"},
            )
        else:
            mask_shape = in_vw.shape
            if any(s is None for s in mask_shape):
                # NOTE: this uses symbolic shape - can be an issue with
                # theano.clone and random numbers
                # https://groups.google.com/forum/#!topic/theano-users/P7Mv7Fg0kUs
                warnings.warn("using symbolic shape for dropout mask, "
                              "which can be an issue with theano.clone")
                mask_shape = in_vw.variable.shape
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            mask = srng.normal(mask_shape, avg=1.0, std=sigma, dtype=fX)
            network.create_vw(
                "default",
                variable=in_vw.variable * mask,
                shape=in_vw.shape,
                tags={"output"},
            )


@core.register_node("spatial_dropout")
class SpatialDropoutNode(core.NodeImpl):

    """
    node that drops out random filters

    Each filter is either on or off.
    """

    hyperparameter_names = ("dropout_probability",
                            "probability",
                            "p",
                            "deterministic")

    def compute_output(self, network, in_vw):
        deterministic = network.find_hyperparameter(["deterministic"])
        p = network.find_hyperparameter(["dropout_probability",
                                         "probability",
                                         "p"],
                                        0)
        if deterministic or p == 0:
            network.copy_vw(
                name="default",
                previous_vw=in_vw,
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
                mask_shape = in_vw.symbolic_shape()
            # FIXME generalize to other shape dimensions.
            # assume this is of the form bc01 (batch, channel, width, height)
            mask_shape = mask_shape[:2]
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            # set bernoulli probability to be inverse of dropout probability
            # because 1 means to keep the unit
            bernoulli_prob = 1 - p
            mask = rescale_factor * srng.binomial(mask_shape,
                                                  p=bernoulli_prob,
                                                  dtype=fX)
            mask = mask.dimshuffle(0, 1, 'x', 'x')
            network.create_vw(
                "default",
                variable=in_vw.variable * mask,
                shape=in_vw.shape,
                tags={"output"},
            )


@core.register_node("gaussian_spatial_dropout")
class GaussianSpatialDropoutNode(core.NodeImpl):

    """
    node that adds gaussian noise to each filters
    """

    hyperparameter_names = ("sigma",
                            "dropout_probability",
                            "probability",
                            "p",
                            "deterministic")

    def compute_output(self, network, in_vw):
        deterministic = network.find_hyperparameter(["deterministic"])
        sigma = network.find_hyperparameter(["sigma"], None)
        if sigma is None:
            p = network.find_hyperparameter(["dropout_probability",
                                             "probability",
                                             "p"],
                                            0)
            if p == 0:
                sigma = 0
            else:
                # derive gaussian dropout variance from bernoulli dropout
                # probability
                sigma = T.sqrt(p / (1 - p))
        if deterministic or sigma == 0:
            network.copy_vw(
                name="default",
                previous_vw=in_vw,
                tags={"output"},
            )
        else:
            mask_shape = in_vw.shape
            if any(s is None for s in mask_shape):
                # NOTE: this uses symbolic shape - can be an issue with
                # theano.clone and random numbers
                # https://groups.google.com/forum/#!topic/theano-users/P7Mv7Fg0kUs
                warnings.warn("using symbolic shape for dropout mask, "
                              "which can be an issue with theano.clone")
                mask_shape = in_vw.symbolic_shape()
            # FIXME generalize to other shape dimensions.
            # assume this is of the form bc01 (batch, channel, width, height)
            mask_shape = mask_shape[:2]
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            mask = srng.normal(mask_shape, avg=1.0, std=sigma, dtype=fX)
            mask = mask.dimshuffle(0, 1, 'x', 'x')
            network.create_vw(
                "default",
                variable=in_vw.variable * mask,
                shape=in_vw.shape,
                tags={"output"},
            )
