"""
from
"Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift"
http://arxiv.org/abs/1502.03167
"""
import warnings

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


# done to match cuDNN
DEFAULT_MOVING_VAR_TYPE = "inv_std"


@treeano.register_node("simple_batch_normalization")
class SimpleBatchNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("epsilon", "inits")

    def _make_param(self, network, in_vw, name, tags):
        return network.create_vw(
            name=name,
            is_shared=True,
            shape=(in_vw.shape[1],),
            tags={"parameter"}.union(tags),
            default_inits=[],
        ).variable.dimshuffle("x", 0, *(["x"] * (in_vw.ndim - 2)))

    def compute_output(self, network, in_vw):
        in_var = in_vw.variable
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        axis = tuple([i for i in range(in_vw.ndim) if i != 1])
        mean = in_var.mean(axis=axis, keepdims=True)
        std = T.sqrt(in_var.var(axis=axis, keepdims=True) + epsilon)
        gamma = self._make_param(network, in_vw, "gamma", {"weight"})
        beta = self._make_param(network, in_vw, "beta", {"bias"})
        network.create_vw(
            name="default",
            # NOTE: 20150907 it is faster to combine gamma + std
            # before broadcasting
            variable=(in_var - mean) * ((gamma + 1) / std) + beta,
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("no_scale_batch_normalization")
class NoScaleBatchNormalizationNode(treeano.NodeImpl):

    # TODO mostly copy-pasted from above

    hyperparameter_names = ("epsilon", "inits")

    def _make_param(self, network, in_vw, name, tags):
        return network.create_vw(
            name=name,
            is_shared=True,
            shape=(in_vw.shape[1],),
            tags={"parameter"}.union(tags),
            default_inits=[],
        ).variable.dimshuffle("x", 0, *(["x"] * (in_vw.ndim - 2)))

    def compute_output(self, network, in_vw):
        in_var = in_vw.variable
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        axis = tuple([i for i in range(in_vw.ndim) if i != 1])
        mean = in_var.mean(axis=axis, keepdims=True)
        std = T.sqrt(in_var.var(axis=axis, keepdims=True) + epsilon)
        beta = self._make_param(network, in_vw, "beta", {"bias"})
        network.create_vw(
            name="default",
            # NOTE: 20150907 it is faster to divide by std before
            # broadcasting than to just divide by std
            variable=(in_var - mean) * (1 / std) + beta,
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("simple_batch_mean_normalization")
class SimpleBatchMeanNormalizationNode(treeano.NodeImpl):

    def _make_param(self, network, in_vw, name, tags):
        return network.create_vw(
            name=name,
            is_shared=True,
            shape=(in_vw.shape[1],),
            tags={"parameter"}.union(tags),
            default_inits=[],
        ).variable.dimshuffle("x", 0, *(["x"] * (in_vw.ndim - 2)))

    def compute_output(self, network, in_vw):
        in_var = in_vw.variable
        axis = tuple([i for i in range(in_vw.ndim) if i != 1])
        mean = in_var.mean(axis=axis, keepdims=True)
        beta = self._make_param(network, in_vw, "beta", {"bias"})
        network.create_vw(
            name="default",
            variable=in_var - mean + beta,
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("advanced_batch_normalization")
class AdvancedBatchNormalizationNode(treeano.NodeImpl):

    # TODO prefix hyperparameters with bn_
    hyperparameter_names = (
        # weight of moving mean/variance put on new minibatches
        "alpha",
        "epsilon",
        "gamma_inits",
        "beta_inits",
        "inits",
        # whether or not moving stats should be used to calculate output
        "bn_use_moving_stats",
        # whether or not moving mean/var should be updated
        "bn_update_moving_stats",
        # which axes should have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "parameter_axes",
        # which axes should not have their own independent parameters
        # only one of parameter_axes and non_parameter_axes should be set
        "non_parameter_axes",
        # which axes should be normalized over
        # only one of normalization_axes and non_normalization_axes should be
        # set
        "normalization_axes",
        # which axes should not be normalized over
        # only one of normalization_axes and non_normalization_axes should be
        # set
        "non_normalization_axes",
        # how to keep moving var
        "moving_var_type",
        # whether or not the mean should be backprop-ed through
        "consider_mean_constant",
        # whether or not the variance should be backprop-ed through
        "consider_var_constant",)

    def compute_output(self, network, in_vw):
        deterministic = network.find_hyperparameter(["deterministic"])

        moving_var_type = network.find_hyperparameter(
            ["moving_var_type"], DEFAULT_MOVING_VAR_TYPE)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)

        if moving_var_type == "log_var":
            moving_var_init_value = 1.0

            def transform_var(v):
                return T.log(v + epsilon)

            def untransform_var(v):
                return T.exp(v)
        elif moving_var_type == "var":
            moving_var_init_value = 0.0

            def transform_var(v):
                return v

            def untransform_var(v):
                return v
        elif moving_var_type == "inv_std":
            moving_var_init_value = 0.0

            def transform_var(v):
                return T.inv(T.sqrt(v) + epsilon)

            def untransform_var(v):
                return T.sqr(T.inv(v))

        # -----------------------------------------------
        # calculate axes to have parameters/normalization
        # -----------------------------------------------

        # axes over which there are parameters for each element
        # ie. parameter_axes == [1, 2] means shape[1] * shape[2] total
        # parameters - one for each combination of shape[1] and shape[2]
        parameter_axes = treeano.utils.find_axes(
            network,
            in_vw.ndim,
            positive_keys=["parameter_axes"],
            negative_keys=["non_parameter_axes"])
        parameter_broadcastable = tuple([idx not in parameter_axes
                                         for idx in range(in_vw.ndim)])
        parameter_shape = tuple([1 if b else s
                                 for b, s in zip(parameter_broadcastable,
                                                 in_vw.shape)])
        # axes to normalize over - ie. subtract the mean across these axes
        normalization_axes = treeano.utils.find_axes(
            network,
            in_vw.ndim,
            positive_keys=["normalization_axes"],
            negative_keys=["non_normalization_axes"])
        stats_shape = tuple([1 if idx in normalization_axes else s
                             for idx, s in enumerate(in_vw.shape)])
        stats_broadcastable = tuple([idx in normalization_axes
                                     for idx in range(in_vw.ndim)])
        assert all([s is not None for s in stats_shape])

        # -----------------------
        # initialize shared state
        # -----------------------

        _gamma = network.create_vw(
            name="gamma",
            is_shared=True,
            shape=parameter_shape,
            tags={"parameter", "weight"},
            # TODO try uniform init between -0.05 and 0.05
            default_inits=[],
            default_inits_hyperparameters=["gamma_inits",
                                           "inits"],
        )
        _beta = network.create_vw(
            name="beta",
            is_shared=True,
            shape=parameter_shape,
            tags={"parameter", "bias"},
            default_inits=[],
            default_inits_hyperparameters=["beta_inits",
                                           "inits"],
        )
        gamma = T.patternbroadcast(_gamma.variable, parameter_broadcastable)
        beta = T.patternbroadcast(_beta.variable, parameter_broadcastable)

        moving_mean = network.create_vw(
            name="mean",
            is_shared=True,
            shape=stats_shape,
            tags={"state"},
            default_inits=[],
        )
        moving_var = network.create_vw(
            name="var",
            is_shared=True,
            shape=stats_shape,
            tags={"state"},
            default_inits=[treeano.inits.ConstantInit(moving_var_init_value)],
        )

        # ------------------------
        # calculate input mean/var
        # ------------------------

        in_mean = T.mean(in_vw.variable,
                         axis=normalization_axes,
                         keepdims=True)
        biased_in_var = T.var(in_vw.variable,
                              axis=normalization_axes,
                              keepdims=True)
        batch_axis = network.find_hyperparameter(["batch_axis"])
        if batch_axis is None:
            in_var = biased_in_var
        else:
            batch_size = in_vw.shape[batch_axis]
            if batch_size is None:
                batch_size = in_vw.variable.shape[batch_axis]
            else:
                batch_size = np.array(batch_size)
            batch_size = batch_size.astype(fX)
            unbias_factor = treeano.utils.as_fX(batch_size / (batch_size - 1))
            in_var = unbias_factor * biased_in_var

        # save the mean/var for updating and debugging
        network.create_vw(
            name="in_mean",
            variable=in_mean,
            tags={},
            shape=stats_shape,
        )
        network.create_vw(
            name="in_var",
            variable=in_var,
            tags={},
            shape=stats_shape,
        )

        # ----------------
        # calculate output
        # ----------------

        bn_use_moving_stats = network.find_hyperparameter(
            ["bn_use_moving_stats"], False)
        if bn_use_moving_stats:
            effective_mean = T.patternbroadcast(moving_mean.variable,
                                                stats_broadcastable)
            effective_var = T.patternbroadcast(
                untransform_var(moving_var.variable),
                stats_broadcastable)
        else:
            if deterministic:
                msg = ("Batch normalization does not use `deterministic` flag"
                       " to control whether or not moving stats are used for"
                       " computation. In this case `bn_use_moving_stats` is "
                       "False, thus per-minibatch stats will be used and may"
                       "be stochastic (depending on how minibatches are"
                       "created), and not only a function of the input"
                       "observation")
                warnings.warn(msg)
            effective_mean = in_mean
            effective_var = in_var

        if network.find_hyperparameter(["consider_mean_constant"], False):
            effective_mean = T.consider_constant(effective_mean)
        if network.find_hyperparameter(["consider_var_constant"], False):
            effective_var = T.consider_constant(effective_var)

        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        denom = T.sqrt(effective_var + epsilon)
        scaled = (in_vw.variable - effective_mean) / denom
        output = (1 + gamma) * scaled + beta
        network.create_vw(
            name="default",
            variable=output,
            shape=in_vw.shape,
            tags={"output"},
        )

    def new_update_deltas(self, network):
        if not network.find_hyperparameter(["bn_update_moving_stats"], True):
            return super(AdvancedBatchNormalizationNode,
                         self).new_update_deltas(network)

        moving_var_type = network.find_hyperparameter(
            ["moving_var_type"], DEFAULT_MOVING_VAR_TYPE)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)

        if moving_var_type == "log_var":
            moving_var_init_value = 1.0

            def transform_var(v):
                return T.log(v + epsilon)

            def untransform_var(v):
                return T.exp(v)
        elif moving_var_type == "var":
            moving_var_init_value = 0.0

            def transform_var(v):
                return v

            def untransform_var(v):
                return v
        elif moving_var_type == "inv_std":
            moving_var_init_value = 0.0

            def transform_var(v):
                return T.inv(T.sqrt(v) + epsilon)

            def untransform_var(v):
                return T.sqr(T.inv(v))

        moving_mean = network.get_vw("mean").variable
        moving_var = network.get_vw("var").variable
        in_mean = network.get_vw("in_mean").variable
        in_var = network.get_vw("in_var").variable
        alpha = network.find_hyperparameter(["alpha"], 0.1)

        updates = [
            (moving_mean, moving_mean * (1 - alpha) + in_mean * alpha),
            (moving_var,
             moving_var * (1 - alpha) + transform_var(in_var) * alpha),
        ]

        return treeano.UpdateDeltas.from_updates(updates)


def BatchNormalizationNode(name, share_filter_weights=True, **kwargs):
    if share_filter_weights:
        kwargs["parameter_axes"] = [1]
    else:
        kwargs["non_parameter_axes"] = [0]
    return AdvancedBatchNormalizationNode(name=name,
                                          non_normalization_axes=[1],
                                          **kwargs)
