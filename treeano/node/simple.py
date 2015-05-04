"""
a bunch of simple, composable nodes - not necessarily simple in their
implementation, but simple in that they do a single thing
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import theano
import theano.tensor as T
from fields import Fields

from .base import Node, WrapperNode


class ReferenceNode(Node, Fields.name.reference):

    """
    provides dependencies into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them
    """

    def init_state(self):
        self.graph.add_dependency(self.reference, self.name)

    def compute_output(self):
        return dict(
            default=self.get_input()
        )


class SequentialNode(WrapperNode, Fields.name.nodes):

    """
    applies several nodes sequentially
    """

    def architecture_children(self):
        return self.nodes

    def init_state(self):
        children_names = self.children_names()
        for from_name, to_name in zip(children_names,
                                      children_names[1:]):
            self.graph.add_dependency(from_name, to_name)
        # set input of first child as default input of this node
        self.forward_input_to(children_names[0])
        # set input of this node as output of final child
        self.take_input_from(children_names[-1])


class ContainerNode(WrapperNode, Fields.name.nodes):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    def architecture_children(self):
        return self.nodes

    def init_state(self):
        # by default, returns the output of its first child
        # ---
        # this was done because it's a sensible default, and other nodes
        # assume that every node has an output
        # additionally, returning the input of this node didn't work, because
        # sometimes the node has no input (eg. if it contains the input node)
        self.take_input_from(self.nodes[0].name)


class HyperparameterNode(WrapperNode, Fields.name.node.hyperparameters):

    """
    for providing hyperparameters to a subtree
    """

    def __init__(self, name, node, **hyperparameters):
        # override init to allow for using keyword arguments
        super(HyperparameterNode, self).__init__(name, node, hyperparameters)

    def architecture_children(self):
        return [self.node]

    def get_hyperparameter(self, hyperparameter_name):
        return self.hyperparameters.get(hyperparameter_name, None)

    def init_state(self):
        self.forward_input_to(self.node.name)
        self.take_input_from(self.node.name)


class InputNode(Node, Fields.name.shape.dtype[theano.config.floatX].broadcastable[None]):

    """
    an entry point into the network
    """

    def compute_output(self):
        self.create_variable(
            name="input_var",
            shape=self.shape,
            dtype=self.dtype,
            broadcastable=self.broadcastable,
            is_shared=False,
            tags=["input"],
        )
        return dict(
            default=self.input_var,
        )


class IdentityNode(Node, Fields.name):

    """
    returns input
    """

    def compute_output(self):
        return dict(
            default=self.get_input()
        )

LOSS_AGGREGATORS = {
    'mean': T.mean,
    'sum': T.sum,
}


class CostNode(Node,
               Fields.name.target_reference.loss_function.loss_aggregator):

    """
    takes in a loss function and a reference to a target node, and computes
    the aggregate loss between the nodes input and the target
    """

    def __init__(self,
                 name,
                 target_reference,
                 loss_function=None,
                 loss_aggregator=None):
        super(CostNode, self).__init__(name,
                                       target_reference,
                                       loss_function,
                                       loss_aggregator)

    def init_state(self):
        self.graph.add_dependency(self.target_reference,
                                  self.name,
                                  to_key="target")

    def compute_output(self):
        preds = self.get_input().variable
        target = self.get_input(to_key="target").variable

        loss_function = self.find_hyperparameter("loss_function")
        self.cost = loss_function(preds, target)

        loss_aggregator = self.find_hyperparameter("loss_aggregator", "mean")
        loss_aggregator = LOSS_AGGREGATORS.get(loss_aggregator,
                                               # allow user defined function
                                               loss_aggregator)
        self.aggregate_cost = loss_aggregator(self.cost)

        self.create_variable(
            "result",
            variable=self.aggregate_cost,
            shape=(),
        )
        return dict(
            default=self.result,
        )


class UpdateScaleNode(WrapperNode, Fields.name.node.scale_factor):

    def architecture_children(self):
        return [self.node]

    def init_state(self):
        self.forward_input_to(self.node.name)
        self.take_input_from(self.node.name)

    def compute_update_deltas(self, update_deltas):
        # FIXME parameterize which nodes to search for (eg. maybe we want
        # to scale state updates)
        parameters = self.find_variables_in_subtree(["parameter"])
        for parameter in parameters:
            print(parameter.variable, update_deltas[parameter.variable])
            update_deltas[parameter.variable] *= self.scale_factor
