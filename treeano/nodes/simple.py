"""
a bunch of simple, composable nodes - not necessarily simple in their
implementation, but simple in that they do a single thing
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import theano
import theano.tensor as T

from .. import core


@core.register_node("reference")
class ReferenceNode(core.NodeImpl):

    """
    provides dependencies into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them
    """

    hyperparameter_names = ("reference",)

    def init_state(self, network):
        network.take_output_from(network.find_hyperparameter(["reference"]))


@core.register_node("sequential")
class SequentialNode(core.WrapperNodeImpl):

    """
    applies several nodes sequentially
    """

    def init_state(self, network):
        # pass input to first child and get output from last child
        super(SequentialNode, self).init_state(network)
        # set dependencies of children sequentially
        children_names = [c.name for c in self.architecture_children()]
        for from_name, to_name in zip(children_names,
                                      children_names[1:]):
            network.add_dependency(from_name, to_name)


@core.register_node("container")
class ContainerNode(core.WrapperNodeImpl):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    input_keys = ("first_child_output",)

    def init_state(self, network):
        # by default, returns the output of its first child
        # ---
        # this was done because it's a sensible default, and other nodes
        # assume that every node has an output
        # additionally, returning the input of this node didn't work, because
        # sometimes the node has no input (eg. if it contains the input
        # node)
        children = self.architecture_children()
        network.take_output_from(children[0].name,
                                 to_key="first_child_output")


@core.register_node("hyperparameter")
class HyperparameterNode(core.Wrapper1NodeImpl):

    """
    for providing hyperparameters to a subtree
    """

    def __init__(self, name, node, **hyperparameters):
        # set hyperparameter keys to be all passed in keys
        self.hyperparameter_names = hyperparameters.keys()
        # override init to allow for using keyword arguments
        super(HyperparameterNode, self).__init__(name, node, **hyperparameters)


@core.register_node("input")
class InputNode(core.NodeImpl):

    """
    an entry point into the network
    """

    hyperparameter_names = ("input_shape",
                            "shape",
                            "input_dtype",
                            "dtype",
                            "input_broadcastable",
                            "broadcastable")
    input_keys = ()

    def compute_output(self, network):
        network.create_variable(
            name="default",
            shape=network.find_hyperparameter(["input_shape",
                                               "shape"]),
            dtype=network.find_hyperparameter(["input_dtype",
                                               "dtype"],
                                              theano.config.floatX),
            broadcastable=network.find_hyperparameter(["input_broadcastable",
                                                       "broadcastable"],
                                                      None),
            is_shared=False,
            tags=["input"],
        )


@core.register_node("identity")
class IdentityNode(core.NodeImpl):

    """
    returns input
    """
    # NOTE: default implementation of NodeImpl is Identity

LOSS_AGGREGATORS = {
    'mean': T.mean,
    'sum': T.sum,
}


@core.register_node("cost")
class CostNode(core.NodeImpl):

    """
    takes in a loss function and a reference to a target node, and computes
    the aggregate loss between the nodes input and the target
    """

    hyperparameter_names = ("target_reference",
                            "reference",
                            "loss_function",
                            "loss_aggregator")
    input_keys = ("default", "target")

    def init_state(self, network):
        network.take_output_from(
            network.find_hyperparameter(["target_reference",
                                         "reference"]),
            to_key="target")

    def compute_output(self, network, preds, target):
        loss_function = network.find_hyperparameter(["loss_function"])

        loss_aggregator = network.find_hyperparameter(["loss_aggregator"],
                                                      "mean")
        loss_aggregator = LOSS_AGGREGATORS.get(loss_aggregator,
                                               # allow user defined function
                                               loss_aggregator)
        cost = loss_function(preds.variable, target.variable)
        aggregate_cost = loss_aggregator(cost)

        network.create_variable(
            "default",
            variable=aggregate_cost,
            shape=(),
        )


@core.register_node("update_scale")
class UpdateScaleNode(core.Wrapper1NodeImpl):

    """
    scales updates from above the tree by multiplying by a constant scale
    factor
    """

    hyperparameter_names = ("update_scale_factor", "scale_factor")

    def mutate_update_deltas(self, network, update_deltas):
        # TODO parameterize which nodes to search for (eg. maybe we want
        # to scale state updates)
        scale_factor = network.find_hyperparameter(["update_scale_factor",
                                                    "scale_factor"])
        parameters = network.find_variables_in_subtree(["parameter"])
        for parameter in parameters:
            update_deltas[parameter.variable] *= scale_factor
