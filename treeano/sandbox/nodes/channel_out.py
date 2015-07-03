"""
from "From Maxout to Channel-Out: Encoding Information on Sparse Pathways"
http://arxiv.org/abs/1312.1909

NOTE: implementation seems quite slow
"""

import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn


@treeano.register_node("channel_out")
class ChannelOutNode(tn.BaseActivationNode):

    hyperparameter_names = ("num_pieces",
                            "feature_pool_axis",
                            "axis")

    def activation(self, network, in_vw):
        # NOTE: mostly copied from FeaturePoolNode
        k = network.find_hyperparameter(["num_pieces"])
        axis = network.find_hyperparameter(
            ["feature_pool_axis",
             "axis"],
            # by default, the first non-batch axis
            treeano.utils.nth_non_batch_axis(network, 0))

        # shape calculation
        in_shape = in_vw.shape
        in_features = in_shape[axis]
        assert (in_features % k) == 0
        out_shape = list(in_shape)
        out_shape[axis] = in_shape[axis] // k
        out_shape = tuple(out_shape)

        # calculate indices of maximum activation
        in_var = in_vw.variable
        symbolic_shape = in_vw.symbolic_shape()
        new_symbolic_shape = (symbolic_shape[:axis]
                              + (out_shape[axis], k) +
                              symbolic_shape[axis + 1:])
        reshaped = in_var.reshape(new_symbolic_shape)
        if True:
            # this implementation seems to be slightly faster
            maxed = T.max(reshaped, axis=axis + 1, keepdims=True)

            mask = T.eq(maxed, reshaped).reshape(symbolic_shape)
        else:
            max_idxs = T.argmax(reshaped, axis=axis + 1, keepdims=True)

            # calculate indices of each unit
            arange_pattern = ["x"] * (in_vw.ndim + 1)
            arange_pattern[axis + 1] = 0
            idxs = T.arange(k).dimshuffle(tuple(arange_pattern))

            mask = T.eq(max_idxs, idxs).reshape(symbolic_shape)
        return in_vw.variable * mask
