"""
from
"Adaptive Normalized Risk-Averting Training For Deep Neural Networks"
http://arxiv.org/abs/1506.02690
"""
import functools

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


# paramters / twists
# ---

# the order of magnitude of lambda seemed to change very slowly, so this is
# an attempt to improve that
# NOTE: this is not part of the paper
ANRAT_USE_LOG_LAMBDA = True

# this is seems to be a stability change because float32 cannot handle e^(10^2)
# NOTE: not in paper, but the authors did something similar under the name
# NRSE
USE_NRSE = False


def _RAE(pred, target, lambda_, p, q, i32_target=False):
    """
    risk-averting error
    RAE[p,q](f, y) = mean(exp(lambda^q * ||f - y||^p))
    """
    assert pred.dtype == fX
    assert pred.ndim == 2
    # note: this is intentionally explicit, instead of just looking at
    # the dtype of target - to make it clearer that this kind of polymorphism
    # can go on
    if i32_target:
        assert target.dtype == "int32"
        assert target.ndim == 1
        # optimization
        # ---
        # take the loss only on the entries with a 1, instead of on each
        # element
        diff = 1 - pred[T.arange(target.shape[0]), target]
        diff_sum_squared = T.sqr(diff)
    else:
        assert target.dtype == fX
        assert target.ndim == 2
        diff = target - pred
        diff_sum_squared = T.sqr(diff).sum(axis=1)

    if p == 2:
        # optimization for p == 2
        # TODO will theano optimize this away?
        norm_raised_to_p = diff_sum_squared
    else:
        norm_raised_to_p = diff_sum_squared ** (p / 2.0)

    if USE_NRSE:
        element_loss = T.exp(-lambda_ ** q * norm_raised_to_p)
    else:
        try:
            lambda_val = lambda_.eval()
        except:
            lambda_val = lambda_
        assert not np.isinf(np.exp(lambda_val ** q, dtype=fX)), dict(
            msg="lambda too high for dtype",
            dtype=fX,
            lambda_val=lambda_val
        )
        element_loss = T.exp(lambda_ ** q * norm_raised_to_p)
    return T.mean(element_loss)


def _NRAE(pred, target, lambda_, p, q, **kwargs):
    """
    normalized risk-averting error
    NRAE[p,q](f,y) = log(RAE[p,q](f,y)) / lambda^q
    """
    res = T.log(_RAE(pred, target, lambda_, p, q, **kwargs)) / (lambda_ ** q)
    if USE_NRSE:
        return -res
    else:
        return res


def _ANRAT(pred, target, lambda_, p, q, r, alpha, **kwargs):
    """
    Adaptive Normalized Risk-Averting Training
    loss(f,y) = NRAE[p,q](f,y) + alpha * ||lambda||^-r
    """
    return (_NRAE(pred, target, lambda_, p, q, **kwargs)
            + alpha * abs(lambda_) ** (-r))


# TODO make NRAE node


@treeano.register_node("anrat")
class ANRATNode(treeano.WrapperNodeImpl):
    children_container = tn.ElementwiseCostNode.children_container
    hyperparameter_names = ("nrae_p",
                            "nrae_q",
                            "anrat_r",
                            "anrat_alpha",
                            "alpha",
                            "anrat_initial_lambda",
                            "i32_target")

    def architecture_children(self):
        return [tn.ElementwiseCostNode(
            self.name + "_elementwise",
            self.raw_children())]

    def init_state(self, network):
        super(ANRATNode, self).init_state(network)

        # setting initial lambda to 5 instead of 10, because 10 is too large
        # for the default parameters
        # TODO might also want to add clipping to cap the value of lambda
        initial_lambda = network.find_hyperparameter(["anrat_initial_lambda"],
                                                     5)
        if ANRAT_USE_LOG_LAMBDA:
            initial_lambda = np.log(initial_lambda)

        lambda_vw = network.create_vw(
            name="lambda",
            is_shared=True,
            shape=(),
            tags={"parameter"},
            default_inits=[treeano.inits.ConstantInit(initial_lambda)],
        )
        p = network.find_hyperparameter(["nrae_p"], 2)
        q = network.find_hyperparameter(["nrae_q"], 2)
        r = network.find_hyperparameter(["anrat_r"], 1)
        alpha = network.find_hyperparameter(["anrat_alpha", "alpha"], 0.1)
        i32_target = network.find_hyperparameter(["i32_target"], False)
        lambda_var = lambda_vw.variable

        if ANRAT_USE_LOG_LAMBDA:
            lambda_var = T.exp(lambda_var)

        cost_function = functools.partial(
            _ANRAT,
            lambda_=lambda_var,
            p=p,
            q=q,
            r=r,
            alpha=alpha,
            i32_target=i32_target)
        network.set_hyperparameter(self.name + "_elementwise",
                                   "cost_function",
                                   cost_function)
