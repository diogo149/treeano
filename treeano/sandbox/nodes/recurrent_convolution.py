"""
from "Recurrent Convolutional Neural Network for Object Recognition"
http://www.xlhu.cn/papers/Liang15-cvpr.pdf
"""

import toolz
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from treeano.sandbox.nodes import lrn


fX = theano.config.floatX


@treeano.register_node("default_recurrent_conv_2d")
class DefaultRecurrentConv2DNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = ("inits",
                            "num_filters",
                            "filter_size",
                            "conv_pad",
                            "pad")
    # TODO parameterize
    steps = 3

    def architecture_children(self):
        # TODO set LRN n = num_filters / 8 + 1
        nodes = [
            # NOTE: not explicitly giving the first conv a pad of "same",
            # since the first conv can have any output shape
            tn.DnnConv2DWithBiasNode(self.name + "_conv0"),
            tn.IdentityNode(self.name + "_z0"),
            tn.ReLUNode(self.name + "_z0_relu"),
            lrn.LocalResponseNormalizationNode(self.name + "_z0_lrn"),
            tn.IdentityNode(self.name + "_x0"),
        ]
        for t in range(1, self.steps + 1):
            nodes += [
                tn.DnnConv2DWithBiasNode(self.name + "_conv%d" % t,
                                         stride=(1, 1),
                                         pad="same"),
                tn.ElementwiseSumNode(
                    self.name + "_sum%d" % t,
                    [tn.ReferenceNode(self.name + "_sum%d_curr" % t,
                                      reference=self.name + "_conv%d" % t),
                     tn.ReferenceNode(self.name + "_sum%d_prev" % t,
                                      reference=self.name + "_z0")]),
                tn.IdentityNode(self.name + "_z%d" % t),
                tn.ReLUNode(self.name + "_z%d_relu" % t),
                lrn.LocalResponseNormalizationNode(self.name + "_z%d_lrn" % t),
                tn.IdentityNode(self.name + "_x%d" % t),
            ]
        return [tn.SequentialNode(self.name + "_sequential", nodes)]

    def init_state(self, network):
        super(DefaultRecurrentConv2DNode, self).init_state(network)
        target_root_node_name = self.name + "_conv1"
        for t in range(2, self.steps + 1):
            root_node_name = self.name + "_conv%d" % t
            inits = [treeano.inits.TiedInit(root_node_name,
                                            target_root_node_name)]
            network.set_hyperparameter(root_node_name,
                                       "inits",
                                       inits)
