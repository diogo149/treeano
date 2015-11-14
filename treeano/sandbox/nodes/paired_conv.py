"""
node for 2 conv's paired together, which allows more flexible combinations of
filter size and padding - specifically even filter sizes can have "same"
padding
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import canopy

fX = theano.config.floatX


@treeano.register_node("paired_conv")
class PairedConvNode(treeano.WrapperNodeImpl):

    hyperparameter_names = ("inits",
                            "filter_size",
                            "num_filters",
                            "conv_pad",
                            "pad")
    children_container = treeano.core.DictChildrenContainerSchema(
        conv=treeano.core.ChildContainer,
        separator=treeano.core.ChildContainer,
    )

    def architecture_children(self):
        children = self.raw_children()
        conv_node = children["conv"]
        separator_node = children["separator"]
        return [tn.SequentialNode(
            self.name + "_sequential",
            [canopy.node_utils.suffix_node(conv_node, "_1"),
             separator_node,
             canopy.node_utils.suffix_node(conv_node, "_2")])]

    def init_state(self, network):
        super(PairedConvNode, self).init_state(network)
        filter_size = network.find_hyperparameter(["filter_size"])
        # calculate effective total filter size
        total_filter_size = tuple([fs * 2 - 1 for fs in filter_size])
        # by default, do same padding
        pad = network.find_hyperparameter(["conv_pad", "pad"], "same")
        total_pad = tn.conv.conv_parse_pad(total_filter_size, pad)
        second_pad = tuple([p // 2 for p in total_pad])
        first_pad = tuple([p - p2 for p, p2 in zip(total_pad, second_pad)])
        conv_node_name = self.raw_children()["conv"].name
        network.set_hyperparameter(conv_node_name + "_1",
                                   "pad",
                                   first_pad)
        network.set_hyperparameter(conv_node_name + "_2",
                                   "pad",
                                   second_pad)
