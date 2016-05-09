"""
from http://arxiv.org/abs/1511.08400
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


@treeano.register_node("feed_forward_norm_stabilizer")
class FeedForwardNormStabilizerNode(treeano.NodeImpl):

    hyperparameter_names = ("cost_reference",
                            "node1_name",
                            "node2_name",
                            "beta",)

    input_keys = ("default", "node1", "node2")

    def init_long_range_dependencies(self, network):
        # forward cost
        network.forward_output_to(
            network.find_hyperparameter(["cost_reference"]),
            from_key="norm_stabilizer_cost",
            to_key=network.find_hyperparameter(
                ["to_key"],
                "feed_forward_norm_stabilizer(%s)" % self.name))

        # take inputs
        network.take_output_from(network.find_hyperparameter(["node1_name"]),
                                 to_key="node1")
        network.take_output_from(network.find_hyperparameter(["node2_name"]),
                                 to_key="node2")

    def compute_output(self, network, in_vw, node1_vw, node2_vw):
        # have output pass through
        super(FeedForwardNormStabilizerNode, self).compute_output(network,
                                                                  in_vw)
        beta = network.find_hyperparameter(["beta"])
        node1_norm = node1_vw.variable.flatten(2).norm(L=2, axis=1)
        node2_norm = node2_vw.variable.flatten(2).norm(L=2, axis=1)
        norm_diff = T.sqr(node1_norm - node2_norm).mean()
        network.create_vw(
            name="norm_stabilizer_cost",
            variable=beta * norm_diff,
            shape=(),
            tags={"monitor"},

        )


@treeano.register_node("constant_norm_stabilizer")
class ConstantNormStabilizerNode(treeano.NodeImpl):

    hyperparameter_names = ("cost_reference",
                            "target_name",
                            "beta",
                            "value",
                            "axis")

    input_keys = ("default", "target")

    def init_long_range_dependencies(self, network):
        # forward cost
        network.forward_output_to(
            network.find_hyperparameter(["cost_reference"]),
            from_key="norm_stabilizer_cost",
            to_key=network.find_hyperparameter(
                ["to_key"],
                "constant_norm_stabilizer(%s)" % self.name))

        # take inputs
        network.take_output_from(network.find_hyperparameter(["target_name"]),
                                 to_key="target")

    def compute_output(self, network, in_vw, target_vw):
        # have output pass through
        super(ConstantNormStabilizerNode, self).compute_output(network,
                                                               in_vw)
        beta = network.find_hyperparameter(["beta"])
        value = network.find_hyperparameter(["value"])
        axis = network.find_hyperparameter(["axis"])
        target_norm = target_vw.variable.norm(L=2, axis=axis)
        norm_diff = T.sqr(target_norm - value).mean()
        network.create_vw(
            name="norm_stabilizer_cost",
            variable=beta * norm_diff,
            shape=(),
            tags={"monitor"},

        )


@treeano.register_node("constant_scaled_norm_stabilizer")
class ConstantScaledNormStabilizerNode(treeano.NodeImpl):

    hyperparameter_names = ("cost_reference",
                            "target_name",
                            "beta",
                            "value",
                            "axis")

    input_keys = ("default", "target")

    def init_long_range_dependencies(self, network):
        # forward cost
        network.forward_output_to(
            network.find_hyperparameter(["cost_reference"]),
            from_key="norm_stabilizer_cost",
            to_key=network.find_hyperparameter(
                ["to_key"],
                "constant_scaled_norm_stabilizer(%s)" % self.name))

        # take inputs
        network.take_output_from(network.find_hyperparameter(["target_name"]),
                                 to_key="target")

    def compute_output(self, network, in_vw, target_vw):
        # have output pass through
        super(ConstantScaledNormStabilizerNode, self).compute_output(network,
                                                                     in_vw)
        beta = network.find_hyperparameter(["beta"])
        value = network.find_hyperparameter(["value"])
        axis = network.find_hyperparameter(["axis"])
        target_norm = target_vw.variable.norm(L=2, axis=axis)
        norm_diff = T.sqr(target_norm / value - 1).mean()
        network.create_vw(
            name="norm_stabilizer_cost",
            variable=beta * norm_diff,
            shape=(),
            tags={"monitor"},

        )
