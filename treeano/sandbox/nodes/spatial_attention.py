import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


@treeano.register_node("spatial_feature_point")
class SpatialFeaturePointNode(treeano.NodeImpl):

    """
    extracts expected feature points from input spatial tensor
    (assumes that the input tensor has been converted into a probability
    distribution over locations - eg. through a spatial softmax)

    based on feature point layer from
    "End-to-End Training of Deep Visuomotor Policies"
    http://arxiv.org/abs/1504.00702

    combined with spatial softmax results in something referred to as
    a "spatial soft argmax"
    """

    def compute_output(self, network, in_vw):
        # assumes dim 0 is batch dim and dim 1 is channel dim
        num_spatial_dims = in_vw.ndim - 2
        # output has shape (batch_size, num_channels, num_dims)
        # so that each channel has it's own expected coordinate
        out_shape = in_vw.shape[:2] + (num_spatial_dims,)

        ss = in_vw.symbolic_shape()
        locations = []
        for idx, axis in enumerate(range(2, in_vw.ndim)):
            # NOTE: from 0 to 1
            raw_locs = treeano.utils.linspace(0, 1, ss[axis])
            padded_shape = tuple([ss[axis] if i == idx else 1
                                  for i in range(num_spatial_dims)])
            repeated = raw_locs.reshape(padded_shape)
            for idx2, axis2 in enumerate(range(2, in_vw.ndim)):
                if idx != idx2:
                    repeated = repeated.repeat(ss[axis2], axis=idx2)
            locations.append(repeated)

        # combine spatial coordinates in new axis by flattening across
        # spatial locations
        combined_locations = T.concatenate(
            [loc.reshape((-1, 1)) for loc in locations],
            axis=1)

        # flatten to 3 axes
        flattened = in_vw.variable.flatten(3)
        # dot product with locations
        out_var = T.dot(flattened, combined_locations)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@treeano.register_node("pairwise_distance")
class PairwiseDistanceNode(treeano.NodeImpl):

    """
    takes in output of spatial feature point node and computes
    pairwise distances for all feature points
    """

    def compute_output(self, network, in_vw):
        assert in_vw.ndim == 3
        in_var = in_vw.variable

        pairwise_diff = (in_var.dimshuffle(0, 1, "x", 2)
                         - in_var.dimshuffle(0, "x", 1, 2))
        pairwise_dist = T.sqrt(T.sqr(pairwise_diff).sum(axis=3) + 1e-8)
        pairwise_dist = pairwise_dist.flatten(2)

        out_shape = (in_vw.shape[0], in_vw.shape[1] ** 2)

        network.create_vw(
            "default",
            variable=pairwise_dist,
            shape=out_shape,
            tags={"output"},
        )


def spatial_soft_argmax(name):
    return tn.SequentialNode(
        name,
        [tn.SpatialSoftmaxNode(name + "_spatial_softmax"),
         SpatialFeaturePointNode(name + "_feature_point")])


def standard_tanh_spatial_attention_2d_node(name,
                                            num_filters,
                                            output_channels=None):
    """
    NOTE: if output_channels is not None, this should be the number
    of input channels
    """
    conv2_filters = 1 if output_channels is None else output_channels

    attention_nodes = [
        tn.Conv2DWithBiasNode(name + "_conv1",
                              filter_size=(1, 1),
                              num_filters=num_filters),
        tn.ScaledTanhNode(name + "_tanh"),
        tn.Conv2DWithBiasNode(name + "_conv2",
                              filter_size=(1, 1),
                              num_filters=conv2_filters),
        tn.SpatialSoftmaxNode(name + "_softmax"),
    ]
    if output_channels is None:
        attention_nodes += [
            tn.AddBroadcastNode(name + "_bcast", axes=(1,)),
        ]

    # multiply input by attention weights and sum across spatial dimensions
    nodes = [
        tn.ElementwiseProductNode(
            name + "_combine",
            [tn.IdentityNode(name + "_input"),
             tn.SequentialNode(
                 name + "_attention",
                 attention_nodes
            )]),
        tn.FlattenNode(name + "_flatten", outdim=3),
        tn.SumNode(name + "_sum", axis=2),
    ]

    return tn.SequentialNode(name, nodes)
