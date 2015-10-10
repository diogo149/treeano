"""
winner-take-all sparsity activations

from "Winner-Take-All Autoencoders"
http://arxiv.org/abs/1409.2752
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

DEFAULT_POPULATION_SPARSITY_LEVEL = 0.05


@treeano.register_node("wta_spatial_sparsity")
class WTASpatialSparsityNode(treeano.NodeImpl):

    """
    keeps only the top activation for each filter map
    """

    def compute_output(self, network, in_vw):

        # TODO parameterize
        # TODO assume spatial axes are 2 and 3
        spatial_axes = (2, 3)

        in_var = in_vw.variable
        threshold = T.max(in_var, axis=spatial_axes, keepdims=True)
        mask = in_var >= threshold
        mask = theano.gradient.disconnected_grad(mask)
        network.create_vw(
            "default",
            variable=mask * in_var,
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("wta_sparsity_node")
class WTASparsityNode(treeano.NodeImpl):

    """
    performs a winner-take-all sparsity activation, with both:
    - population sparsity:
      only keeps the top k% of values for each channel across each batch
    - spatial sparsity:
      if there are extra axes (neither channel nor batch), takes the max
      across the extra axes

    ie. if minibatch has 100 entries, for each channel location, zero out all
    but top k over each minibatch
    """

    hyperparameter_names = ("percentile",
                            "channel_axis")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        k = network.find_hyperparameter(["percentile"],
                                        DEFAULT_POPULATION_SPARSITY_LEVEL)
        channel_axis = network.find_hyperparameter(["channel_axis"], None)
        if channel_axis is None:
            channel_axis = treeano.utils.nth_non_batch_axis(network, 0)
        # TODO parameterize
        batch_axis = network.find_hyperparameter(["batch_axis"])

        # remove "extra" axes
        # TODO add parameter for ignored axes
        remaining_axes = list(range(in_vw.ndim))
        remaining_axes.remove(batch_axis)
        remaining_axes.remove(channel_axis)
        in_var = in_vw.variable
        if remaining_axes:
            batch_and_channel = T.max(in_var,
                                      axis=remaining_axes,
                                      keepdims=True)
        else:
            batch_and_channel = in_var

        # find threshold across batch
        percentile = (1 - k) * 100
        threshold = treeano.theano_extensions.tensor.percentile(
            batch_and_channel,
            percentile,
            axis=batch_axis,
            keepdims=True)

        # generate output
        mask = in_var >= threshold
        if remaining_axes:
            # need to zero out extra axes as well
            # not just based on population sparsity
            mask *= (in_var >= batch_and_channel)
        mask = theano.gradient.disconnected_grad(mask)
        network.create_vw(
            "default",
            variable=mask * in_var,
            shape=in_vw.shape,
            tags={"output"},
        )
