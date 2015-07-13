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
import treeano.lasagne.nodes as tl

import canopy


@treeano.register_node("inception")
class InceptionNode(treeano.WrapperNodeImpl):

    children_container = treeano.core.DictChildrenContainerSchema(
        activation=treeano.core.ChildContainer,
    )

    def architecture_children(self):
        activation_container = self._children.get("activation")
        if activation_container is None:
            activation = tn.ReLUNode(self.name + "_relu")
        else:
            activation = activation_container.children

        path_1x1 = tn.SequentialNode(
            self.name + "_1x1",
            [tl.Conv2DDNNNode(
                self.name + "_1x1conv",
                filter_size=(1, 1),
                border_mode="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_1x1")])
        path_3x3 = tn.SequentialNode(
            self.name + "_3x3",
            [tl.Conv2DDNNNode(
                self.name + "_3x3reduce",
                filter_size=(1, 1),
                border_mode="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_3x3reduce"),
             tl.Conv2DDNNNode(
                self.name + "_3x3conv",
                filter_size=(3, 3),
                border_mode="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_3x3")])
        path_5x5 = tn.SequentialNode(
            self.name + "_5x5",
            [tl.Conv2DDNNNode(
                self.name + "_5x5reduce",
                filter_size=(1, 1),
                border_mode="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_5x5reduce"),
             tl.Conv2DDNNNode(
                self.name + "_5x5conv",
                filter_size=(5, 5),
                border_mode="same"),
             canopy.node_utils.format_node_name(activation,
                                                self.name + "_%s_5x5")])
        path_pool = tn.SequentialNode(
            self.name + "_poolproj",
            [tl.MaxPool2DDNNNode(
                self.name + "_poolprojmax",
                pool_stride=(1, 1),
                # TODO parameterize
                # also need to make padding be dependent on pool size
                pool_size=(3, 3),
                pad=(1, 1)),
             tl.Conv2DDNNNode(
                self.name + "_poolproj1x1",
                filter_size=(1, 1),
                border_mode="same"),
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
