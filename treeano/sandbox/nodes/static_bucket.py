from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node('static_bucket')
class StaticBucketNode(treeano.NodeImpl):

    """
    node that uses soft attention over a specified number of buckets for
    regression. attention for each output dimension is shared in the structured
    case and independent in the independent case
    """

    hyperparameter_names = ("num_buckets",
                            "bucket_type",
                            "output_dim")

    def compute_output(self, network, in_vw):
        num_buckets = network.find_hyperparameter(["num_buckets"])
        bucket_type = network.find_hyperparameter(["bucket_type"],
                                                  'structured')
        assert bucket_type in {'independent', 'structured'}
        output_dim = network.find_hyperparameter(["output_dim"])

        in_var = in_vw.variable
        input_shape = in_vw.shape
        # TODO: make this work for other input shapes

        assert len(input_shape) == 2
        output_shape = (input_shape[0], output_dim)

        if bucket_type == 'structured':
            attention_W = network.create_vw(
                name="attention_weight",
                is_shared=True,
                shape=(input_shape[-1], num_buckets),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

            attention_b = network.create_vw(
                name="attention_bias",
                is_shared=True,
                shape=(num_buckets,),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable

            buckets_W = network.create_vw(
                name="buckets_weight",
                is_shared=True,
                shape=(num_buckets, output_dim),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

            z = T.dot(in_var, attention_W) + attention_b.dimshuffle('x', 0)
            attention = treeano.utils.stable_softmax(z, axis=1)
            out_var = T.dot(attention, buckets_W)

        elif bucket_type == 'independent':
            attention_W = network.create_vw(
                name="attention_weight",
                is_shared=True,
                shape=(output_dim, input_shape[-1], num_buckets),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

            attention_b = network.create_vw(
                name="attention_bias",
                is_shared=True,
                shape=(output_dim, num_buckets),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable

            buckets_W = network.create_vw(
                name="buckets_weight",
                is_shared=True,
                shape=(output_dim, num_buckets),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

            z = T.dot(in_var, attention_W) + attention_b.dimshuffle('x', 0, 1)
            attention = treeano.utils.stable_softmax(z, axis=2)
            out_var = (attention * buckets_W.dimshuffle('x', 0, 1)).sum(axis=2)

        network.create_vw(
            name="default",
            variable=out_var,
            shape=output_shape,
            tags={"output"},
        )
