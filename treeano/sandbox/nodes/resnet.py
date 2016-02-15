import functools
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy
from treeano.sandbox.nodes import batch_normalization as bn

fX = theano.config.floatX


@treeano.register_node("strided_downsample")
class StridedDownsampleNode(treeano.NodeImpl):

    hyperparameter_names = ("strides",)

    def compute_output(self, network, in_vw):
        strides = network.find_hyperparameter(["strides"])

        out_slices = []
        out_shape = list(in_vw.shape)
        for idx, stride in enumerate(strides):
            out_slices.append(slice(None, None, stride))
            size = out_shape[idx]
            if size is not None:
                out_shape[idx] = (size + stride - 1) // stride

        network.create_vw(
            "default",
            variable=in_vw.variable[tuple(out_slices)],
            shape=tuple(out_shape),
            tags={"output"},
        )


@treeano.register_node("resnet_init_conv_2d")
class ResnetInitConv2DNode(treeano.NodeImpl):

    """
    NOTE: originally copy-pasted from Conv2DNode
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

        pad = tn.conv.conv_parse_pad(filter_size, pad)

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

        # calculate identity for resnet init
        # ---
        # read hyperparams
        identity_ratio = network.find_hyperparameter(["identity_ratio"], 0.5)
        ratio_on_input = network.find_hyperparameter(["ratio_on_input"], True)
        # find center spatial location
        dim0_idx = filter_shape[2] // 2
        dim1_idx = filter_shape[3] // 2
        # create identity kernel
        ratio_idx = 1 if ratio_on_input else 0
        init = np.zeros(filter_shape, dtype=theano.config.floatX)
        for i in range(min(filter_shape[0],
                           filter_shape[1],
                           int(identity_ratio * filter_shape[ratio_idx]))):
            init[i, i, dim0_idx, dim1_idx] += 1

        out_var = T.nnet.conv2d(input=in_vw.variable,
                                filters=W + init,
                                input_shape=in_vw.shape,
                                filter_shape=filter_shape,
                                border_mode=pad,
                                subsample=stride)

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


@treeano.register_node("resnet_init_conv_2d_with_bias")
class ResnetInitConv2DWithBiasNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = ResnetInitConv2DNode.hyperparameter_names

    def architecture_children(self):
        return [
            tn.SequentialNode(
                self._name + "_sequential",
                [ResnetInitConv2DNode(self._name + "_conv"),
                 tn.AddBiasNode(self._name + "_bias",
                                broadcastable_axes=(0, 2, 3))])]


def residual_block_conv_2d(name,
                           num_filters,
                           num_layers,
                           increase_dim=None,
                           conv_node=tn.Conv2DNode,
                           bn_node=bn.BatchNormalizationNode,
                           activation_node=tn.ReLUNode,
                           input_num_filters=None,
                           projection_filter_size=(1, 1),
                           increase_dim_stride=(2, 2),
                           no_identity=False):
    assert num_layers >= 2

    if increase_dim is not None:
        assert increase_dim in {"projection", "pad"}
        first_stride = increase_dim_stride
        if increase_dim == "projection":
            identity_node = tn.SequentialNode(
                name + "_projection",
                [conv_node(name + "_projectionconv",
                           num_filters=num_filters,
                           filter_size=projection_filter_size,
                           stride=first_stride,
                           pad="same"),
                 bn_node(name + "_projectionbn"),
                 # TODO try w/ relu
                 ])
        elif increase_dim == "pad":
            assert input_num_filters is not None
            identity_node = tn.SequentialNode(
                name + "_pad",
                [StridedDownsampleNode(
                    name + "_stride",
                    strides=(1, 1) + first_stride),
                 tn.PadNode(
                     name + "_addpad",
                     padding=(0, (num_filters - input_num_filters) // 2, 0, 0))])
    else:
        first_stride = (1, 1)
        identity_node = tn.IdentityNode(name + "_identity")

    nodes = []
    # first node
    for i in range(num_layers):
        if i == 0:
            # first conv
            # ---
            # same as middle convs, but with stride
            nodes += [
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          stride=first_stride,
                          pad="same"),
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
            ]
        elif i == num_layers - 1:
            # last conv
            # ---
            # same as middle convs, but no activation
            nodes += [
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          pad="same"),
                bn_node(name + "_bn%d" % i),
            ]
        else:
            nodes += [
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          pad="same"),
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
            ]
    residual_node = tn.SequentialNode(name + "_seq", nodes)

    if no_identity:
        # ability to disable resnet connections
        return residual_node
    else:
        return tn.ElementwiseSumNode(name,
                                     [identity_node,
                                      residual_node])


def resnet_init_block_conv_2d(*args, **kwargs):
    return residual_block_conv_2d(*args,
                                  conv_node=ResnetInitConv2DNode,
                                  **kwargs)


def pool_with_projection_2d(name,
                            projection_filters,
                            stride=(2, 2),
                            filter_size=(3, 3),
                            bn_node=bn.BatchNormalizationNode):

    pool_node = tn.MaxPool2DNode(name + "_pool",
                                 pool_size=stride,
                                 stride=stride)

    projection_node = tn.SequentialNode(
        name + "_projection",
        [tn.Conv2DNode(name + "_projectionconv",
                       num_filters=projection_filters,
                       filter_size=filter_size,
                       stride=stride,
                       pad="same"),
         bn_node(name + "_projectionbn"),
         # TODO try w/ relu
         ])

    return tn.ConcatenateNode(name, [pool_node, projection_node])
