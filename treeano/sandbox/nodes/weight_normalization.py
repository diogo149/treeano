"""
from "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"
http://arxiv.org/abs/1602.07868
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
import treeano
import treeano.nodes as tn
import canopy

fX = theano.config.floatX


@treeano.register_node("weight_normalized_dnn_conv_2d")
class WeightNormalizedDnnConv2DNode(treeano.NodeImpl):

    """
    node for 2D convolution
    """

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_stride",
                            "stride",
                            "conv_pad",
                            "pad",
                            "conv_mode")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"], (1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], (0, 0))
        pad = tn.conv.conv_parse_pad(filter_size, pad)
        # by default, do convolution instead of cross-correlation
        # rationale: be compatible with standard (non-cuDNN) conv2d
        conv_mode = network.find_hyperparameter(["conv_mode"], "conv")
        assert len(filter_size) == 2
        assert conv_mode in ["conv", "cross"]

        # create weight
        num_channels = in_vw.shape[1]
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=(num_filters, num_channels) + tuple(filter_size),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable

        norm = T.sqrt(T.sqr(W).flatten(2).sum(axis=1))

        g = network.create_vw(
            name="scale",
            is_shared=True,
            shape=(num_filters, ),
            tags={"parameter"},
            default_inits=[treeano.inits.ConstantInit(norm.eval())],
        ).variable

        new_weight = W * (g / norm).dimshuffle(0, "x", "x", "x")

        out_var = dnn.dnn_conv(img=in_vw.variable,
                               kerns=new_weight,
                               border_mode=pad,
                               subsample=stride,
                               conv_mode=conv_mode)

        out_shape = tn.conv.conv_output_shape(input_shape=in_vw.shape,
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


@treeano.register_node("weight_normalized_dnn_conv_2d_with_bias")
class WeightNormalizedDnnConv2DWithBiasNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = WeightNormalizedDnnConv2DNode.hyperparameter_names

    def architecture_children(self):
        return [
            tn.SequentialNode(
                self._name + "_sequential",
                [WeightNormalizedDnnConv2DNode(self._name + "_conv"),
                 # TODO add hyperparameter untie_biases
                 tn.AddBiasNode(self._name + "_bias",
                                broadcastable_axes=(0, 2, 3))])]


@treeano.register_node("scaled_init_weight_normalized_dnn_conv_2d_with_bias")
class ScaledInitWeightNormalizedDnnConv2DWithBiasNode(treeano.NodeImpl):

    """
    node for 2D convolution
    """

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_stride",
                            "stride",
                            "conv_pad",
                            "pad",
                            "conv_mode")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"], (1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], (0, 0))
        pad = tn.conv.conv_parse_pad(filter_size, pad)
        # by default, do convolution instead of cross-correlation
        # rationale: be compatible with standard (non-cuDNN) conv2d
        conv_mode = network.find_hyperparameter(["conv_mode"], "conv")
        assert len(filter_size) == 2
        assert conv_mode in ["conv", "cross"]

        # create weight
        num_channels = in_vw.shape[1]
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=(num_filters, num_channels) + tuple(filter_size),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable

        norm = T.sqrt(T.sqr(W).flatten(2).sum(axis=1))

        no_g = dnn.dnn_conv(img=in_vw.variable,
                            kerns=W / norm.dimshuffle(0, "x", "x", "x"),
                            border_mode=pad,
                            subsample=stride,
                            conv_mode=conv_mode)

        in_shape = list(in_vw.shape)
        for idx in range(len(in_shape)):
            if in_shape[idx] is None:
                in_shape[idx] = 256
        sample_data = np.array(no_g.eval(
            {in_vw.variable: np.random.randn(*in_shape).astype(fX)}))
        mu = sample_data.std(axis=(0, 2, 3))
        sigma = sample_data.std(axis=(0, 2, 3))

        g = network.create_vw(
            name="scale",
            is_shared=True,
            shape=(num_filters, ),
            tags={"parameter"},
            default_inits=[treeano.inits.ConstantInit(1. / sigma)],
        ).variable

        b = network.create_vw(
            name="bias",
            is_shared=True,
            shape=(num_filters, ),
            tags={"parameter", "bias"},
            default_inits=[treeano.inits.ConstantInit(-mu / sigma)],
        ).variable

        new_weight = W * (g / norm).dimshuffle(0, "x", "x", "x")

        out_var = (dnn.dnn_conv(img=in_vw.variable,
                                kerns=new_weight,
                                border_mode=pad,
                                subsample=stride,
                                conv_mode=conv_mode) +
                   b.dimshuffle("x", 0, "x", "x"))

        out_shape = tn.conv.conv_output_shape(input_shape=in_vw.shape,
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
