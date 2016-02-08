"""
convolutional nodes
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

from .. import core


def conv_output_length(input_size, conv_size, stride, pad):
    """
    calculates the output size along a single axis for a conv operation
    """
    if input_size is None:
        return None
    without_stride = input_size + 2 * pad - conv_size + 1
    # equivalent to np.ceil(without_stride / stride)
    output_size = (without_stride + stride - 1) // stride
    return output_size


def conv_output_shape(input_shape,
                      num_filters,
                      axes,
                      conv_shape,
                      strides,
                      pads):
    """
    compute output shape for a conv
    """
    output_shape = list(input_shape)
    assert 1 not in axes
    output_shape[1] = num_filters
    for axis, conv_size, stride, pad in zip(axes,
                                            conv_shape,
                                            strides,
                                            pads):
        output_shape[axis] = conv_output_length(input_shape[axis],
                                                conv_size,
                                                stride,
                                                pad)
    return tuple(output_shape)


def conv_parse_pad(filter_size, pad):
    if pad == "valid":
        return (0,) * len(filter_size)
    elif pad == "full":
        return tuple([x - 1 for x in filter_size])
    elif pad in ("same", "half"):
        new_pad = []
        for f in filter_size:
            assert f % 2
            new_pad += [f // 2]
        return tuple(new_pad)
    else:
        assert len(pad) == len(filter_size)
        return pad


@core.register_node("conv_2d")
class Conv2DNode(core.NodeImpl):

    """
    node for 2D convolution
    """

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_stride",
                            "stride",
                            "conv_pad",
                            "pad")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"], (1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], "valid")

        pad = conv_parse_pad(filter_size, pad)

        # HACK figure out if this is necessary
        # convert numerical pad to valid or full
        # if pad == (0, 0):
        #     pad = "valid"
        # elif pad == tuple([fs - 1 for fs in filter_size]):
        #     pad = "full"
        # assert pad in ["valid", "full"]
        assert len(filter_size) == 2

        # create weight
        num_channels = in_vw.shape[1]
        filter_shape = (num_filters, num_channels) + tuple(filter_size)
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=filter_shape,
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable

        out_var = T.nnet.conv2d(input=in_vw.variable,
                                filters=W,
                                input_shape=in_vw.shape,
                                filter_shape=filter_shape,
                                border_mode=pad,
                                subsample=stride)

        out_shape = conv_output_shape(input_shape=in_vw.shape,
                                      num_filters=num_filters,
                                      axes=(2, 3),
                                      conv_shape=filter_size,
                                      strides=stride,
                                      pads=pad)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("conv_3d")
class Conv3DNode(core.NodeImpl):

    """
    node for 3D convolution
    """

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_stride",
                            "stride",
                            "conv_pad",
                            "pad",
                            "include_bias")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"],
                                             (1, 1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], "valid")
        include_bias = network.find_hyperparameter(["include_bias"], False)
        assert len(filter_size) == 3
        assert pad == "valid"

        # create weight
        num_channels = in_vw.shape[1]
        filter_shape = (num_filters, num_channels) + tuple(filter_size)
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=filter_shape,
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable
        # create bias
        if include_bias:
            b = network.create_vw(
                name="bias",
                is_shared=True,
                shape=(num_filters,),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable
        else:
            b = T.zeros(num_filters)

        from theano.tensor.nnet.Conv3D import conv3D
        # conv3D takes V in order: (batch, row, column, time, in channel)
        # and W in order: (out channel, row, column, time ,in channel)
        # but we keep the dimensions that W is stored in consistent with other
        # convolutions, so we have to dimshuffle here
        out_var = conv3D(V=in_vw.variable.dimshuffle(0, 2, 3, 4, 1),
                         W=W.dimshuffle(0, 2, 3, 4, 1),
                         b=b,
                         d=stride)

        out_shape = conv_output_shape(input_shape=in_vw.shape,
                                      num_filters=num_filters,
                                      axes=(2, 3, 4),
                                      conv_shape=filter_size,
                                      strides=stride,
                                      pads=(0, 0, 0))

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("conv_3d2d")
class Conv3D2DNode(core.NodeImpl):

    """
    performs 3D convolution via 2D convolution
    see: theano.tensor.nnet.conv3d2d.conv3d
    """

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_stride",
                            "stride",
                            "conv_pad",
                            "pad")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"],
                                             (1, 1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], "valid")
        assert len(filter_size) == 3
        assert pad == "valid"
        assert stride == (1, 1, 1)

        # create weight
        num_channels = in_vw.shape[1]
        filter_shape = (num_filters, num_channels) + tuple(filter_size)
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=filter_shape,
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable

        from theano.tensor.nnet.conv3d2d import conv3d
        # takes signals in order: (batch, time, channels, row, column)
        # and filters in order: (out channel, time, in channels, row, column)
        # but we keep the dimensions that W is stored in consistent with other
        # convolutions, so we have to dimshuffle here
        order = (0, 2, 1, 3, 4)
        out_var = conv3d(signals=in_vw.variable.dimshuffle(*order),
                         filters=W.dimshuffle(*order),
                         signals_shape=[in_vw.shape[o] for o in order],
                         filters_shape=[filter_shape[o] for o in order],
                         # HACK as of 20150916, conv3d does a check
                         # if isinstance(border_mode, str), so we manually
                         # cast as a string
                         border_mode=str("valid"))

        out_shape = conv_output_shape(input_shape=in_vw.shape,
                                      num_filters=num_filters,
                                      axes=(2, 3, 4),
                                      conv_shape=filter_size,
                                      strides=stride,
                                      pads=conv_parse_pad(filter_size, pad))

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
