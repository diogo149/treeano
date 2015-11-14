"""
from
"Going Deeper with Convolutions"
http://arxiv.org/abs/1409.4842

WARNING: can be quite slow
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

import canopy


@treeano.register_node("inception")
class InceptionNode(treeano.WrapperNodeImpl):

    children_container = treeano.core.DictChildrenContainerSchema(
        activation=treeano.core.ChildContainer,
    )
    hyperparameter_names = ("num_filters_1x1",
                            "num_filters_3x3reduce",
                            "num_filters_3x3",
                            "num_filters_5x5reduce",
                            "num_filters_5x5",
                            "num_filters_poolproj")

    def architecture_children(self):
        children = self.raw_children()
        if "activation" in children:
            activation = children["activation"]
        else:
            activation = tn.ReLUNode(self.name + "_relu")

        path_1x1 = tn.SequentialNode(
            self.name + "_1x1",
            [tn.DnnConv2DWithBiasNode(
                self.name + "_1x1conv",
                filter_size=(1, 1),
                pad="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_1x1")])
        path_3x3 = tn.SequentialNode(
            self.name + "_3x3",
            [tn.DnnConv2DWithBiasNode(
                self.name + "_3x3reduce",
                filter_size=(1, 1),
                pad="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_3x3reduce"),
             tn.DnnConv2DWithBiasNode(
                self.name + "_3x3conv",
                filter_size=(3, 3),
                pad="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_3x3")])
        path_5x5 = tn.SequentialNode(
            self.name + "_5x5",
            [tn.DnnConv2DWithBiasNode(
                self.name + "_5x5reduce",
                filter_size=(1, 1),
                pad="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_5x5reduce"),
             tn.DnnConv2DWithBiasNode(
                self.name + "_5x5conv",
                filter_size=(5, 5),
                pad="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_5x5")])
        path_pool = tn.SequentialNode(
            self.name + "_poolproj",
            [tn.DnnMaxPoolNode(
                self.name + "_poolprojmax",
                pool_stride=(1, 1),
                # TODO parameterize
                # also need to make padding be dependent on pool size
                pool_size=(3, 3),
                pad=(1, 1)),
             tn.DnnConv2DWithBiasNode(
                self.name + "_poolproj1x1",
                filter_size=(1, 1),
                pad="same"),
             canopy.node_utils.format_node_name(
                 activation, self.name + "_%s_poolproj1x1")])

        return [tn.ConcatenateNode(
            self.name + "_concat",
            [path_1x1,
             path_3x3,
             path_5x5,
             path_pool])]

    def init_state(self, network):
        super(InceptionNode, self).init_state(network)
        network.forward_hyperparameter(self.name + "_1x1conv",
                                       "num_filters",
                                       ["num_filters_1x1"])
        network.forward_hyperparameter(self.name + "_3x3reduce",
                                       "num_filters",
                                       ["num_filters_3x3reduce"])
        network.forward_hyperparameter(self.name + "_3x3conv",
                                       "num_filters",
                                       ["num_filters_3x3"])
        network.forward_hyperparameter(self.name + "_5x5reduce",
                                       "num_filters",
                                       ["num_filters_5x5reduce"])
        network.forward_hyperparameter(self.name + "_5x5conv",
                                       "num_filters",
                                       ["num_filters_5x5"])
        network.forward_hyperparameter(self.name + "_poolproj1x1",
                                       "num_filters",
                                       ["num_filters_poolproj"])
