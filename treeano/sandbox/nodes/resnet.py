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


@treeano.register_node("zero_last_axis_partition")
class _ZeroLastAxisPartitionNode(treeano.NodeImpl):

    """
    zeros out a fraction of a tensor
    """

    hyperparameter_names = ("zero_ratio",
                            "axis")

    def compute_output(self, network, in_vw):
        zero_ratio = network.find_hyperparameter(["zero_ratio"])
        axis = network.find_hyperparameter(["axis"], 1)

        in_var = in_vw.variable
        size = treeano.utils.as_fX(in_var.shape[axis])
        num_zeros = T.round(zero_ratio * size).astype("int32")
        idxs = [None] * (axis - 1) + [slice(-num_zeros, None)]
        out_var = T.set_subtensor(in_var[idxs], 0)

        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )


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
    if increase_dim is not None:
        assert increase_dim in {"projection", "pad"}
        first_stride = increase_dim_stride
        if increase_dim == "projection":
            identity_node = tn.SequentialNode(
                name + "_projection",
                [tn.Conv2DNode(name + "_projectionconv",
                               num_filters=num_filters,
                               filter_size=projection_filter_size,
                               stride=first_stride,
                               pad="same"),
                 bn_node(name + "_projectionbn")])
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
        else:
            nodes += [
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          stride=(1, 1),
                          pad="same"),
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
            ]
    # for last conv, remove activation
    nodes.pop()

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


def resnet_init_projection_conv_2d(name,
                                   num_filters,
                                   num_layers,
                                   bn_node=bn.BatchNormalizationNode,
                                   activation_node=tn.ReLUNode,
                                   stride=(1, 1)):
    nodes = []
    # first node
    for i in range(num_layers):
        if i == 0:
            # first conv
            # ---
            # same as middle convs, but with stride
            nodes += [
                tn.Conv2DNode(name + "_conv%d" % i,
                              num_filters=num_filters,
                              stride=stride,
                              pad="same"),
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
            ]
        else:
            nodes += [
                ResnetInitConv2DNode(name + "_conv%d" % i,
                                     num_filters=num_filters,
                                     stride=(1, 1),
                                     pad="same"),
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
            ]
    # for last conv, remove activation
    nodes.pop()

    return tn.SequentialNode(name + "_seq", nodes)


def preactivation_residual_block_conv_2d(name,
                                         num_filters,
                                         num_layers,
                                         increase_dim=None,
                                         initial_block=False,
                                         conv_node=tn.Conv2DNode,
                                         bn_node=bn.BatchNormalizationNode,
                                         activation_node=tn.ReLUNode,
                                         input_num_filters=None,
                                         projection_filter_size=(1, 1),
                                         increase_dim_stride=(2, 2),
                                         no_identity=False):
    """
    from http://arxiv.org/abs/1603.05027
    """
    if increase_dim is not None:
        assert increase_dim in {"projection", "pad"}
        first_stride = increase_dim_stride
        if increase_dim == "projection":
            # TODO remove pre-activation when initial block
            assert not initial_block
            identity_node = tn.SequentialNode(
                name + "_projection",
                [
                    bn_node(name + "_projectionbn"),
                    activation_node(name + "_projectionactivation"),
                    tn.Conv2DNode(name + "_projectionconv",
                                  num_filters=num_filters,
                                  filter_size=projection_filter_size,
                                  stride=first_stride,
                                  pad="same"),
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
            # maybe remove initial activation
            if not initial_block:
                nodes += [
                    bn_node(name + "_bn%d" % i),
                    activation_node(name + "_activation%d" % i),
                ]
            # same as middle convs, but with stride
            nodes += [
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          stride=first_stride,
                          pad="same"),
            ]
        else:
            nodes += [
                bn_node(name + "_bn%d" % i),
                activation_node(name + "_activation%d" % i),
                conv_node(name + "_conv%d" % i,
                          num_filters=num_filters,
                          stride=(1, 1),
                          pad="same"),
            ]

    residual_node = tn.SequentialNode(name + "_seq", nodes)

    if no_identity:
        # ability to disable resnet connections
        return residual_node
    else:
        return tn.ElementwiseSumNode(name,
                                     [identity_node,
                                      residual_node])


def generalized_residual(name, nodes, identity_ratio=0.5):
    return tn.ElementwiseSumNode(
        name,
        [_ZeroLastAxisPartitionNode(name + "_zero",
                                    zero_ratio=(1 - identity_ratio)),
         tn.SequentialNode(
             name + "_seq",
             nodes)])


def generalized_residual_conv_2d(name,
                                 num_filters,
                                 include_preactivation=True,
                                 bn_node=bn.BatchNormalizationNode,
                                 activation_node=tn.ReLUNode,
                                 conv_node=tn.Conv2DNode,
                                 identity_ratio=0.5):
    """
    generalized resnet block based on pre-activation resnet
    """
    nodes = []
    if include_preactivation:
        # add pre-activation
        nodes += [
            bn_node(name + "_bn"),
            activation_node(name + "_activation"),
        ]
    nodes += [conv_node(name + "_conv", num_filters=num_filters)]
    return generalized_residual(name, nodes, identity_ratio)


def generalized_residual_block_conv_2d(name,
                                       num_filters,
                                       num_layers,
                                       increase_dim=None,
                                       initial_block=False,
                                       bn_node=bn.BatchNormalizationNode,
                                       activation_node=tn.ReLUNode,
                                       conv_node=tn.Conv2DNode,
                                       identity_ratio=0.5,
                                       input_num_filters=None,
                                       projection_filter_size=(1, 1),
                                       increase_dim_stride=(2, 2),
                                       no_identity=False):
    if no_identity:  # HACK for compatibility
        identity_ratio = 0
    nodes = []
    if increase_dim is not None:
        if increase_dim == "projection":
            # TODO remove pre-activation when initial block
            assert not initial_block
            # TODO maybe reduce layers by 1 to have same depth
            # num_layers -= 1
            nodes += [tn.SequentialNode(
                name + "_projection",
                [bn_node(name + "_projectionbn"),
                 activation_node(name + "_projectionactivation"),
                 tn.Conv2DNode(name + "_projectionconv",
                               num_filters=num_filters,
                               filter_size=projection_filter_size,
                               stride=increase_dim_stride,
                               pad="same")])]
        elif increase_dim == "pad":
            assert input_num_filters is not None
            nodes += [tn.SequentialNode(
                name + "_pad",
                [StridedDownsampleNode(
                    name + "_stride",
                    strides=(1, 1) + increase_dim_stride),
                 tn.PadNode(
                     name + "_addpad",
                     padding=(0, (num_filters - input_num_filters) // 2, 0, 0))])]
        else:
            raise ValueError(increase_dim)
    for i in range(num_layers):
        include_preactivation = (not initial_block) or (i != 0)
        nodes += [generalized_residual_conv_2d(
            "%s_%d" % (name, i),
            include_preactivation=include_preactivation,
            num_filters=num_filters,
            activation_node=activation_node,
            identity_ratio=identity_ratio)]
    return tn.SequentialNode(name, nodes)


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
         bn_node(name + "_projectionbn")])

    return tn.ConcatenateNode(name, [pool_node, projection_node])


def forget_gate_conv_2d_node(name,
                             num_filters,
                             filter_size=(3, 3),
                             initial_bias=0):
    return tn.ElementwiseProductNode(
        name,
        [tn.IdentityNode(name + "_identity"),
         tn.SequentialNode(
             name + "_forget",
             [tn.Conv2DWithBiasNode(name + "_conv",
                                    num_filters=num_filters,
                                    filter_size=filter_size,
                                    stride=(1, 1),
                                    pad="same"),
              tn.AddConstantNode(name + "_initial_bias", value=initial_bias),
              tn.SigmoidNode(name + "_sigmoid")])])
