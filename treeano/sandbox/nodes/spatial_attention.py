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
