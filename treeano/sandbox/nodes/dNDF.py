"""
from "Deep Neural Decision Forests"
http://research.microsoft.com/apps/pubs/default.aspx?id=255952
"""

import warnings

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import treeano
import treeano.nodes as tn
from treeano.theano_extensions import tree_probability


@treeano.register_node("probability_linear_combination")
class ProbabilityLinearCombinationNode(treeano.NodeImpl):

    """
    takes inputs as weights for a weighted linear combination of probability
    tensors
    """

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


@treeano.register_node("select_one_along_axis")
class SelectOneAlongAxisNode(treeano.NodeImpl):

    """
    given an axis, selects one tensor along that axis, or
    if deterministic == True, returns the average of tensors along
    that axis

    use case:
    - creating a final layer that is a decision forest, but only training
      one random tree at a time at train time
    """

    hyperparameter_names = ("axis", "deterministic")

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        deterministic = network.find_hyperparameter(["deterministic"], False)

        # calculate output shape
        output_shape = list(in_vw.shape)
        output_shape.pop(axis)

        if deterministic:
            out_var = in_vw.variable.mean(axis=axis)
        else:
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            if in_vw.shape[axis] is None:
                # NOTE: this uses symbolic shape - can be an issue with
                # theano.clone and random numbers
                # https://groups.google.com/forum/#!topic/theano-users/P7Mv7Fg0kUs
                warnings.warn("using symbolic shape for random variable size "
                              "which can be an issue with theano.clone")
            idx = T.argmax(srng.normal([in_vw.symbolic_shape()[axis]]))
            slices = tuple([slice(None) for _ in range(axis)] + [idx])
            out_var = in_vw.variable[slices]

        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(output_shape),
            tags={"output"},
        )


@treeano.register_node("theano_split_probabilities_to_leaf_probabilities")
class TheanoSplitProbabilitiesToLeafProbabilitiesNode(treeano.NodeImpl):

    """
    convert tensor of probabilities of going left at each node of a decision
    tree into a tensor of probabilities of landing at each of the leafs

    this node is implemented in pure Theano
    """

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

    """
    convert tensor of probabilities of going left at each node of a decision
    tree into a tensor of probabilities of landing at each of the leafs

    this node is implemented in numpy and should be more memory efficient than
    the Theano version, and compile faster, but performed in Python (on CPU)
    """

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
