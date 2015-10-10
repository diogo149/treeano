"""
from
"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
http://arxiv.org/abs/1406.4729
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from theano.tensor.signal import downsample

fX = theano.config.floatX


def spp_max_pool_axis_kwargs(in_shape, out_shape):
    symbolic = (treeano.utils.is_variable(in_shape)
                or treeano.utils.is_variable(out_shape))
    # maxpool requires static shape
    assert not symbolic
    if symbolic:
        int_ceil = lambda x: T.ceil(x).astype("int32")
    else:
        int_ceil = lambda x: int(np.ceil(x))

    # eg. if input is 5 and output is 2, each pool size should be 3
    pool_size = int_ceil(in_shape / out_shape)
    # stride should equal pool_size, since we want non-overlapping regions
    stride = pool_size
    # pad as much as possible, since ignore_border=True
    padding = int_ceil((pool_size * out_shape - in_shape) / 2)

    if not symbolic:
        assert padding < pool_size

    return dict(
        ds=pool_size,
        st=stride,
        padding=padding,
    )


def spp_max_pool_kwargs(in_shape, out_shape):
    assert len(in_shape) == len(out_shape)
    axis_res = []
    for i, o in zip(in_shape, out_shape):
        axis_res.append(spp_max_pool_axis_kwargs(i, o))
    return dict(
        ds=tuple([r["ds"] for r in axis_res]),
        st=tuple([r["st"] for r in axis_res]),
        padding=tuple([r["padding"] for r in axis_res]),
        # must be set to true for padding to work
        ignore_border=True,
    )


@treeano.register_node("spatial_pyramid_pooling")
class SpatialPyramidPoolingNode(treeano.NodeImpl):

    """
    eg.
    SpatialPyramidPoolingNode("spp", spp_levels=[(1, 1), (2, 2), (4, 4)])
    """

    hyperparameter_names = ("spp_levels",)

    def compute_output(self, network, in_vw):
        spp_levels = network.find_hyperparameter(["spp_levels"])
        # FIXME generalize to other shape dimensions.
        # assume this is of the form bc01 (batch, channel, width, height)

        # shape calculation
        in_shape = in_vw.symbolic_shape()
        if in_vw.shape[1] is None:
            out_shape1 = None
        else:
            out_shape1 = in_vw.shape[1] * sum(d1 * d2 for d1, d2 in spp_levels)
        out_shape = (in_vw.shape[0], out_shape1)

        # compute out
        mp_kwargs_list = [spp_max_pool_kwargs(in_shape[2:], spp_level)
                          for spp_level in spp_levels]
        pooled = [downsample.max_pool_2d(in_vw.variable, **kwargs)
                  for kwargs in mp_kwargs_list]
        out_var = T.concatenate([p.flatten(2) for p in pooled], axis=1)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
