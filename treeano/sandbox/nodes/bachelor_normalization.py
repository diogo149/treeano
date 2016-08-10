import theano
import theano.tensor as T
import treeano
from treeano.sandbox.nodes import bttf_mean

fX = theano.config.floatX


@treeano.register_node("bachelor_normalization")
class BachelorNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("bttf_alpha",
                            "alpha",
                            "epsilon",
                            "normalization_axes",
                            "update_averages",
                            "deterministic")

    def compute_output(self, network, in_vw):
        alpha = network.find_hyperparameter(["bttf_alpha", "alpha"], 0.95)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-4)
        normalization_axes = network.find_hyperparameter(["normalization_axes"],
                                                         (1,))
        # HACK: using "deterministic" to mean test time
        deterministic = network.find_hyperparameter(["deterministic"], False)
        update_averages = network.find_hyperparameter(["update_averages"],
                                                      not deterministic)

        alpha = treeano.utils.as_fX(alpha)

        if update_averages:
            backprop_to_the_future_mean = bttf_mean.backprop_to_the_future_mean_with_updates
        else:
            backprop_to_the_future_mean = bttf_mean.backprop_to_the_future_mean_no_updates

        state_shape = tuple([in_vw.shape[axis] for axis in normalization_axes])
        state_pattern = ["x"] * in_vw.ndim
        for idx, axis in enumerate(normalization_axes):
            state_pattern[axis] = idx

        def make_state(name, tags, default_inits=None):
            if default_inits is None:
                default_inits = []
            return network.create_vw(
                name=name,
                is_shared=True,
                shape=state_shape,
                tags=tags,
                default_inits=default_inits,
            ).variable

        gamma = make_state("gamma", {"parameter", "weight"})
        beta = make_state("beta", {"parameter", "bias"})
        # mean of input
        mean = make_state("mean", {"state"})
        # gradient of mean of input
        mean_grad = make_state("mean_grad", {"state"})
        # mean of input^2
        squared_mean = make_state("squared_mean", {"state"},
                                  # initializing to 1, so that std = 1
                                  default_inits=[treeano.inits.ConstantInit(1.)])
        # gradient of mean of input^2
        squared_mean_grad = make_state("squared_mean_grad", {"state"})

        in_var = in_vw.variable
        mean_axes = tuple([axis for axis in range(in_var.ndim)
                           if axis not in normalization_axes])
        batch_mean = in_var.mean(axis=mean_axes)
        squared_batch_mean = T.sqr(in_var).mean(axis=mean_axes)

        # expectation of input (x)
        E_x = backprop_to_the_future_mean(batch_mean,
                                          mean,
                                          mean_grad,
                                          alpha)
        # TODO try mixing batch mean with E_x
        # expectation of input squared
        E_x_squared = backprop_to_the_future_mean(squared_batch_mean,
                                                  squared_mean,
                                                  squared_mean_grad,
                                                  alpha)

        # HACK mixing batch and rolling means
        # E_x = 0.5 * E_x + 0.5 * batch_mean
        # E_x_squared = 0.5 * E_x_squared + 0.5 * squared_batch_mean

        if 1:
            mu = E_x
            sigma = T.sqrt(E_x_squared - T.sqr(E_x) + epsilon)

            mu = mu.dimshuffle(state_pattern)
            sigma = sigma.dimshuffle(state_pattern)
            gamma = gamma.dimshuffle(state_pattern)
            beta = beta.dimshuffle(state_pattern)

        else:
            # HACK mixing current value
            E_x = E_x.dimshuffle(state_pattern)
            E_x_squared = E_x_squared.dimshuffle(state_pattern)
            gamma = gamma.dimshuffle(state_pattern)
            beta = beta.dimshuffle(state_pattern)

            E_x = 0.1 * in_var + 0.9 * E_x
            E_x_squared = 0.1 * T.sqr(in_var) + 0.9 * E_x_squared

            mu = E_x
            sigma = T.sqrt(E_x_squared - T.sqr(E_x) + epsilon)

        if 0:
            # HACK don't backprop through sigma
            sigma = T.consider_constant(sigma)

        if 1:
            # HACK using batch mean
            mu = batch_mean
            mu = mu.dimshuffle(state_pattern)

        if 0:
            # HACK using batch variance
            sigma = T.sqrt(in_var.var(axis=mean_axes) + epsilon)
            sigma = sigma.dimshuffle(state_pattern)

        out_var = (in_var - mu) * (T.exp(gamma) / sigma) + beta

        network.create_vw(
            name="default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )

        if 1:
            # HACK monitoring state
            network.create_vw(
                name="mu_mean",
                variable=mu.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="sigma_mean",
                variable=sigma.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="gamma_mean",
                variable=gamma.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="beta_mean",
                variable=beta.mean(),
                shape=(),
                tags={"monitor"},

            )


@treeano.register_node("bachelor_normalization2")
class BachelorNormalization2Node(treeano.NodeImpl):

    hyperparameter_names = ("bttf_alpha",
                            "alpha",
                            "epsilon",
                            "normalization_axes",
                            "update_averages",
                            "deterministic")

    def compute_output(self, network, in_vw):
        alpha = network.find_hyperparameter(["bttf_alpha", "alpha"], 0.95)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-4)
        normalization_axes = network.find_hyperparameter(["normalization_axes"],
                                                         (1,))
        # HACK: using "deterministic" to mean test time
        deterministic = network.find_hyperparameter(["deterministic"], False)
        update_averages = network.find_hyperparameter(["update_averages"],
                                                      not deterministic)

        alpha = treeano.utils.as_fX(alpha)

        if update_averages:
            backprop_to_the_future_mean = bttf_mean.backprop_to_the_future_mean_with_updates
        else:
            backprop_to_the_future_mean = bttf_mean.backprop_to_the_future_mean_no_updates

        state_shape = tuple([in_vw.shape[axis] for axis in normalization_axes])
        state_pattern = ["x"] * in_vw.ndim
        for idx, axis in enumerate(normalization_axes):
            state_pattern[axis] = idx

        def make_state(name, tags, default_inits=None):
            if default_inits is None:
                default_inits = []
            return network.create_vw(
                name=name,
                is_shared=True,
                shape=state_shape,
                tags=tags,
                default_inits=default_inits,
            ).variable

        gamma = make_state("gamma", {"parameter", "weight"})
        beta = make_state("beta", {"parameter", "bias"})
        # mean of input
        mean = make_state("mean", {"state"})
        # gradient of mean of input
        mean_grad = make_state("mean_grad", {"state"})
        var_state_mean = make_state("var_state_mean", {"state"},
                                    # initializing to 1, so that std = 1
                                    default_inits=[treeano.inits.ConstantInit(1.)])
        var_state_mean_grad = make_state("var_state_mean_grad", {"state"})

        in_var = in_vw.variable
        mean_axes = tuple([axis for axis in range(in_var.ndim)
                           if axis not in normalization_axes])
        batch_mean = in_var.mean(axis=mean_axes)

        # expectation of input (x)
        E_x = backprop_to_the_future_mean(batch_mean,
                                          mean,
                                          mean_grad,
                                          alpha)
        # TODO try mixing batch mean with E_x
        if 1:
            batch_var_state = 1. / T.sqrt(in_var.var(axis=mean_axes) + epsilon)
            var_state = backprop_to_the_future_mean(batch_var_state,
                                                    var_state_mean,
                                                    var_state_mean_grad,
                                                    alpha)
            inv_std = var_state

        # HACK mixing batch and rolling means
        # E_x = 0.5 * E_x + 0.5 * batch_mean
        # E_x_squared = 0.5 * E_x_squared + 0.5 * squared_batch_mean

        mu = E_x

        mu = mu.dimshuffle(state_pattern)
        inv_std = inv_std.dimshuffle(state_pattern)
        gamma = gamma.dimshuffle(state_pattern)
        beta = beta.dimshuffle(state_pattern)

        out_var = (in_var - mu) * (T.exp(gamma) * inv_std) + beta

        network.create_vw(
            name="default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )

        if 1:
            # HACK monitoring state
            network.create_vw(
                name="mu_mean",
                variable=mu.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="var_state_effective_mean",
                variable=var_state.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="gamma_mean",
                variable=gamma.mean(),
                shape=(),
                tags={"monitor"},

            )
            network.create_vw(
                name="beta_mean",
                variable=beta.mean(),
                shape=(),
                tags={"monitor"},

            )
