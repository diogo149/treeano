"""
based on "Empirical Bernstein Bounds and Sample Variance Penalization"
http://arxiv.org/abs/0907.3740
and http://www.machinedlearnings.com/2015/11/sample-variance-penalization.html
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def sample_variance_penalty_aggregator(costs,
                                       kappa=0.25,
                                       penalty_type="per_sample"):
    if costs.ndim < 1:
        assert False

    if penalty_type == "per_sample":
        # convert to 1 cost per sample
        if costs.ndim > 1:
            if costs.ndim > 2:
                costs = T.flatten(costs, 2)
            costs = costs.mean(axis=1)
    elif penalty_type == "per_element":
        # leave it as it is
        pass
    else:
        raise ValueError("incorrect penalty_type: {}".format(penalty_type))

    return costs.mean() + kappa * costs.std()


def SampleVariancePenalizationNode(*args, **kwargs):
    # TODO convert to node that takes in appropriate hyperparameters
    assert "aggregator" not in kwargs
    kwargs["aggregator"] = sample_variance_penalty_aggregator
    return tn.TotalCostNode(*args, **kwargs)
