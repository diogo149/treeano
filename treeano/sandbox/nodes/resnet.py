import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy
from treeano.sandbox.nodes import batch_normalization as bn

fX = theano.config.floatX


def residual_block_conv_2d(name,
                           num_filters,
                           num_layers,
                           # TODO add flag for filter size
                           # TODO add flag for bottleneck
                           increase_dim=None,
                           bn_type="default"):
    assert num_layers >= 2

    # TODO add different bn nodes
    assert bn_type == "default"
    bn_node = bn.BatchNormalizationNode

    if increase_dim is not None:
        assert increase_dim in {"projection", "pad"}
        first_stride = (2, 2)
        if increase_dim == "projection":
            identity_node = tn.SequentialNode(
                name + "_projection",
                [
                    tn.Conv2DNode(name + "_projectionconv",
                                  num_filters=num_filters,
                                  # TODO try 1x1 or 3x3
                                  # filter_size=(3, 3),
                                  stride=first_stride,
                                  pad="same"),
                    # TODO try w/o bn
                    bn_node(name + "_projectionbn"),
                ])
        elif increase_dim == "pad":
            assert False
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
                tn.Conv2DNode(name + "_conv%d" % i,
                              num_filters=num_filters,
                              stride=first_stride,
                              pad="same"),
                bn_node(name + "_bn%d" % i),
                tn.ReLUNode(name + "_activtion%d" % i),
            ]
        elif i == num_layers - 1:
            # last conv
            # ---
            # same as middle convs, but no activation
            nodes += [
                tn.Conv2DNode(name + "_conv%d" % i,
                              num_filters=num_filters,
                              pad="same"),
                bn_node(name + "_bn%d" % i),
            ]
        else:
            nodes += [
                tn.Conv2DNode(name + "_conv%d" % i,
                              num_filters=num_filters,
                              pad="same"),
                bn_node(name + "_bn%d" % i),
                tn.ReLUNode(name + "_activtion%d" % i),
            ]
    residual_node = tn.SequentialNode(name + "_seq", nodes)

    return tn.ElementwiseSumNode(name,
                                 [identity_node,
                                  residual_node])
