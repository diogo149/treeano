import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


def no_mean_softmax(x, axis=1):
    """
    similar to stable softmax, but using mean instead of max
    """
    # TODO test performance on axis
    # if this way is slow, could reshape, do the softmax, then reshape back
    if axis == tuple(range(1, x.ndim)):
        # reshape, do softmax, then reshape back, in order to be differentiable
        # TODO could do reshape trick for any set of sequential axes
        # that end with last (eg. 2,3), not only when starting with axis 1
        return no_mean_softmax(x.flatten(2)).reshape(x.shape)
    else:
        e_x = T.exp(x - x.mean(axis=axis, keepdims=True))
        out = e_x / e_x.sum(axis=axis, keepdims=True)
        return out


@treeano.register_node("no_mean_softmax")
class NoMeanSoftmaxNode(tn.BaseActivationNode):
    hyperparameter_names = ("axis",)

    def activation(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], 1)
        return no_mean_softmax(in_vw.variable, axis=axis)


@treeano.register_node("no_batch_normalization")
class NoBatchNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("current_mean_weight",
                            "current_var_weight",
                            "rolling_mean_rate",
                            "rolling_var_rate",
                            # "moving_var_type",
                            # "var_combine_type",
                            "epsilon",
                            "normalization_axes")

    def compute_output(self, network, in_vw):
        current_mean_weight = network.find_hyperparameter(
            ["current_mean_weight"])
        current_var_weight = network.find_hyperparameter(["current_var_weight"])
        # TODO parameterize
        # moving_var_type = network.find_hyperparameter(["moving_var_type"],
        #                                               "var_mean")
        # var_combine_type = network.find_hyperparameter(["var_combine_type"],
        #                                                "var_mean")
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        normalization_axes = network.find_hyperparameter(["normalization_axes"],
                                                         (1,))
        state_shape = tuple([in_vw.shape[axis] for axis in normalization_axes])

        def make_state(name, tags, default_inits=None):
            if default_inits is None:
                default_inits = []
            pattern = ["x"] * in_vw.ndim
            for idx, axis in enumerate(normalization_axes):
                pattern[axis] = idx
            return network.create_vw(
                name=name,
                is_shared=True,
                shape=state_shape,
                tags=tags,
                default_inits=default_inits,
            ).variable.dimshuffle(pattern)

        # parameters
        # initialize gamma to 0 for L2 regularization (1 is added later)
        gamma = make_state("gamma", {"parameter", "weight"})
        beta = make_state("beta", {"parameter", "bias"})
        # exponential moving average of mean/var
        mean_ema = make_state("mean_ema", {"state"})
        var_ema = make_state("var_ema",
                             {"state"},
                             # TODO parameterize for different types
                             # of moving var
                             default_inits=[treeano.inits.ConstantInit(1.0)])

        in_var = in_vw.variable
        in_axes = tuple([i
                         for i in range(in_var.ndim)
                         # exclude batch dimension
                         if (i != 0) and i not in normalization_axes])
        in_mu = in_var.mean(axis=in_axes, keepdims=True)
        in_sigma2 = T.sqr(in_var - mean_ema).mean(axis=in_axes,
                                                  keepdims=True)
        mu = (current_mean_weight * in_mu
              + (1 - current_mean_weight) * mean_ema)
        sigma2 = (current_var_weight * in_sigma2
                  + (1 - current_var_weight) * var_ema)

        # HACK
        update_axes = tuple([i
                             for i in range(in_var.ndim)
                             # include batch dimension
                             if i not in normalization_axes])
        hacky_mu = in_var.mean(axis=update_axes, keepdims=True)
        # TODO see difference between hacky mu and mu
        # TODO try consider_constant on hacky_mu
        # TODO try consider_constant on hacky_mu and combine with current
        # activation
        # TODO try unbiased rolling averages?
        mu = hacky_mu

        out_var = T.exp(gamma) * (in_var - mu) / T.sqrt(sigma2 + epsilon) + beta

        # FIXME
        network.create_vw(
            name="default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )

        # save the mean/var for updating and debugging
        update_axes = tuple([i
                             for i in range(in_var.ndim)
                             # include batch dimension
                             if i not in normalization_axes])
        network.create_vw(
            name="in_mean",
            variable=in_var.mean(axis=update_axes),
            tags={},
            shape=state_shape,
        )
        network.create_vw(
            name="in_var",
            variable=in_var.var(axis=update_axes),
            tags={},
            shape=state_shape,
        )

    def new_update_deltas(self, network):
        rolling_mean_rate = network.find_hyperparameter(["rolling_mean_rate"])
        rolling_var_rate = network.find_hyperparameter(["rolling_var_rate"])

        mean_ema = network.get_vw("mean_ema").variable
        var_ema = network.get_vw("var_ema").variable
        in_mean = network.get_vw("in_mean").variable
        in_var = network.get_vw("in_var").variable

        new_mean = (rolling_mean_rate * mean_ema
                    + (1 - rolling_mean_rate) * in_mean)
        new_var = (rolling_var_rate * var_ema
                   + (1 - rolling_var_rate) * in_var)
        updates = [
            (mean_ema, new_mean),
            (var_ema, new_var),
        ]

        return treeano.UpdateDeltas.from_updates(updates)
