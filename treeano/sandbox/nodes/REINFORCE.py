import warnings
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


@treeano.register_node("normal_sample")
class NormalSampleNode(treeano.NodeImpl):

    input_keys = ("mu", "sigma")
    hyperparameter_names = ("deterministic",)

    def compute_output(self, network, mu_vw, sigma_vw):
        deterministic = network.find_hyperparameter(["deterministic"], False)
        if deterministic:
            res = mu_vw.variable
        else:
            # TODO look at shape of both mu and sigma
            shape = mu_vw.shape
            if any(s is None for s in shape):
                # NOTE: this uses symbolic shape - can be an issue with
                # theano.clone and random numbers
                # https://groups.google.com/forum/#!topic/theano-users/P7Mv7Fg0kUs
                warnings.warn("using symbolic shape for random number shape, "
                              "which can be an issue with theano.clone")
                shape = mu_vw.variable.shape
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            res = srng.normal(shape,
                              avg=mu_vw.variable,
                              std=sigma_vw.variable,
                              dtype=fX)
        network.create_vw(
            "default",
            variable=theano.gradient.disconnected_grad(res),
            shape=mu_vw.shape,
            tags={"output"},
        )


@treeano.register_node("normal_REINFORCE")
class NormalREINFORCECostNode(treeano.NodeImpl):

    """
    cost node to implement REINFORCE algorithm

    include_baseline: whether or not to include a baseline network
    backprop_baseline: whether or not to backprop the baseline update to
                       the rest of the network
    """

    hyperparameter_names = ("include_baseline",
                            "backprop_baseline")
    input_keys = ("state", "mu", "sigma", "reward", "sampled")

    def compute_output(self,
                       network,
                       state_vw,
                       mu_vw,
                       sigma_vw,
                       reward_vw,
                       sampled_vw):
        # want state to have dim (batch size x size of state)
        assert state_vw.ndim == 2
        # want mu to have dim (batch size x number of actions)
        assert mu_vw.ndim == 2

        state = state_vw.variable
        mu = mu_vw.variable
        sigma = sigma_vw.variable
        reward = reward_vw.variable
        sampled = sampled_vw.variable

        # create reward baseline
        bias = network.create_vw(
            name="bias",
            is_shared=True,
            shape=(),
            tags={"parameter", "bias"},
            default_inits=[],
        ).variable
        weight = network.create_vw(
            name="weight",
            is_shared=True,
            shape=(state_vw.shape[1],),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable
        if not network.find_hyperparameter(["backprop_baseline"], False):
            state = theano.gradient.disconnected_grad(state)
        baseline = ((weight.dimshuffle("x", 0) * state).sum(axis=1)
                    + bias)
        if not network.find_hyperparameter(["include_baseline"], True):
            # to try REINFORCE without the baseline network
            baseline = baseline * 0
        # TODO monitor baseline
        constant_baseline = theano.gradient.disconnected_grad(baseline)

        # 1 / (sigma * sqrt(2 * pi)) * exp(-1/2 * ((t - mu) / sigma)^2)
        normal_pdf = (1 / (sigma * treeano.utils.as_fX(np.sqrt(2 * np.pi)))
                      * T.exp(-0.5 * T.sqr((sampled - mu) / sigma)))
        log_normal_pdf = T.log(normal_pdf)
        R = reward - constant_baseline
        # take sum of log pdf
        reinforce_cost = -(R * log_normal_pdf.sum(axis=1)).sum()
        # TODO add parameter as weight for baseline
        baseline_cost = T.sum((reward - baseline) ** 2)

        network.create_vw(
            name="default",
            # variable=reinforce_cost,
            variable=reinforce_cost + baseline_cost,
            shape=(),
            tags={"output", "monitor"},
        )
