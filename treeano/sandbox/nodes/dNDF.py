"""
from "Deep Neural Decision Forests"
http://research.microsoft.com/apps/pubs/default.aspx?id=255952
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.theano_extensions import tree_probability


@treeano.register_node("probability_linear_combination")
class ProbabilityLinearCombinationNode(treeano.NodeImpl):

    hyperparameter_names = ("num_units",
                            "inits")

    def compute_output(self, network, in_vw):
        num_units = network.find_hyperparameter(["num_units"])

        # raw vectors
        pre_softmax_shape = (in_vw.shape[-1], num_units)
        pre_softmax = network.create_vw(
            name="pre_softmax",
            is_shared=True,
            shape=pre_softmax_shape,
            tags={"parameter"},
            default_inits=[],
        ).variable

        # convert to probability vectors
        pi = T.nnet.softmax(pre_softmax)

        output_shape = tuple(in_vw.shape[:-1]) + (num_units, )
        network.create_vw(
            name="default",
            variable=T.dot(in_vw.variable, pi),
            shape=output_shape,
            tags={"output"},
        )


def is_power_of_2(num):
    # bit arithmetic check
    return (num > 0) and ((num & (num - 1)) == 0)


@treeano.register_node("theano_split_probabilities_to_leaf_probabilities")
class TheanoSplitProbabilitiesToLeafProbabilitiesNode(treeano.NodeImpl):

    hyperparameter_names = ()

    def compute_output(self, network, in_vw):
        # calculate output shape
        output_shape = list(in_vw.shape)
        output_shape[1] += 1
        output_shape = tuple(output_shape)

        output_ss = list(in_vw.symbolic_shape())
        output_ss[1] += 1

        left_probs = in_vw.variable
        right_probs = 1 - left_probs

        num_splits = in_vw.shape[1]

        probs = T.ones(output_ss)
        for idx, (l, m, r) in enumerate(
                tree_probability.size_to_tree(num_splits)):
            l_subtensor = probs[:, l:m]
            probs = T.set_subtensor(
                l_subtensor,
                # add new axis so that probabilities broadcast
                l_subtensor * treeano.utils.newaxis(left_probs[:, idx], 1))
            r_subtensor = probs[:, m:r]
            probs = T.set_subtensor(
                r_subtensor,
                # add new axis so that probabilities broadcast
                r_subtensor * treeano.utils.newaxis(right_probs[:, idx], 1))

        network.create_vw(
            "default",
            variable=probs,
            shape=output_shape,
            tags={"output"},
        )


@treeano.register_node("numpy_split_probabilities_to_leaf_probabilities")
class NumpySplitProbabilitiesToLeafProbabilitiesNode(treeano.NodeImpl):

    hyperparameter_names = ()

    def compute_output(self, network, in_vw):
        # calculate output shape
        output_shape = list(in_vw.shape)
        output_shape[1] += 1
        output_shape = tuple(output_shape)

        network.create_vw(
            "default",
            variable=tree_probability.tree_probability(in_vw.variable),
            shape=output_shape,
            tags={"output"},
        )


# NOTE: setting the numpy version as the default, because the
# theano version is very inefficent for large trees
SplitProbabilitiesToLeafProbabilitiesNode \
    = NumpySplitProbabilitiesToLeafProbabilitiesNode
