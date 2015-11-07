"""
nodes based on cuDNN
http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
"""

from theano.sandbox.cuda import dnn

from .. import core
from . import downsample
from . import conv
from . import containers
from . import simple


@core.register_node("dnn_pool")
class DnnPoolNode(core.NodeImpl):

    """
    2D or 3D pooling node that takes in a specified "mode" and utilizes cuDNN
    """

    hyperparameter_names = ("mode",
                            "pool_size",
                            "pool_stride",
                            "stride",
                            "pool_pad",
                            "pad")

    def compute_output(self, network, in_vw):
        mode = network.find_hyperparameter(["mode"])
        pool_size = network.find_hyperparameter(["pool_size"])
        dim = len(pool_size)
        # works for sizes 2 and 3
        assert dim in [2, 3]
        stride = network.find_hyperparameter(["pool_stride",
                                              "stride"],
                                             None)
        if stride is None:
            stride = pool_size
        pad = network.find_hyperparameter(["pool_pad", "pad"], (0,) * dim)
        assert dim == len(stride) == len(pad)
        if dim == 2:
            pool_axes = (2, 3)
        elif dim == 3:
            pool_axes = (2, 3, 4)
        out_shape = downsample.pool_output_shape(
            input_shape=in_vw.shape,
            axes=pool_axes,
            pool_shape=pool_size,
            strides=stride,
            pads=pad)
        out_var = dnn.dnn_pool(img=in_vw.variable,
                               ws=pool_size,
                               stride=stride,
                               pad=pad,
                               mode=mode)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


def DnnMeanPoolNode(*args, **kwargs):
    """
    NOTE: this average does not include padding
    """
    return DnnPoolNode(*args, mode="average_exc_pad", **kwargs)


def DnnMaxPoolNode(*args, **kwargs):
    return DnnPoolNode(*args, mode="max", **kwargs)


@core.register_node("dnn_conv_2d")
class DnnConv2DNode(core.NodeImpl):

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
        pad = conv.conv_parse_pad(filter_size, pad)
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

        out_var = dnn.dnn_conv(img=in_vw.variable,
                               kerns=W,
                               border_mode=pad,
                               subsample=stride,
                               conv_mode=conv_mode)

        out_shape = conv.conv_output_shape(input_shape=in_vw.shape,
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


@core.register_node("dnn_conv_3d")
class DnnConv3DNode(core.NodeImpl):

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
                            "conv_mode")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        num_filters = network.find_hyperparameter(["num_filters"])
        filter_size = network.find_hyperparameter(["filter_size"])
        stride = network.find_hyperparameter(["conv_stride", "stride"],
                                             (1, 1, 1))
        pad = network.find_hyperparameter(["conv_pad", "pad"], (0, 0, 0))
        pad = conv.conv_parse_pad(filter_size, pad)
        # by default, do convolution instead of cross-correlation
        # rationale: be compatible with standard (non-cuDNN) conv2d
        conv_mode = network.find_hyperparameter(["conv_mode"], "conv")
        assert len(filter_size) == 3
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

        out_var = dnn.dnn_conv3d(img=in_vw.variable,
                                 kerns=W,
                                 border_mode=pad,
                                 subsample=stride,
                                 conv_mode=conv_mode)

        out_shape = conv.conv_output_shape(input_shape=in_vw.shape,
                                           num_filters=num_filters,
                                           axes=(2, 3, 4),
                                           conv_shape=filter_size,
                                           strides=stride,
                                           pads=pad)
        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("dnn_conv_2d_with_bias")
class DnnConv2DWithBiasNode(core.Wrapper0NodeImpl):

    hyperparameter_names = DnnConv2DNode.hyperparameter_names

    def architecture_children(self):
        return [
            containers.SequentialNode(
                self._name + "_sequential",
                [DnnConv2DNode(self._name + "_conv"),
                 # TODO add hyperparameter untie_biases
                 simple.AddBiasNode(self._name + "_bias",
                                    broadcastable_axes=(0, 2, 3))])]


@core.register_node("dnn_conv_3d_with_bias")
class DnnConv3DWithBiasNode(core.Wrapper0NodeImpl):

    hyperparameter_names = DnnConv3DNode.hyperparameter_names

    def architecture_children(self):
        return [
            containers.SequentialNode(
                self._name + "_sequential",
                [DnnConv3DNode(self._name + "_conv"),
                 # TODO add hyperparameter untie_biases
                 simple.AddBiasNode(self._name + "_bias",
                                    broadcastable_axes=(0, 2, 3, 4))])]
