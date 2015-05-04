import numpy as np
from fields import Fields
import lasagne

from .base import Node, WrapperNode
from ..update_deltas import UpdateDeltas


class FullyConnectedNode(Node, Fields.name.num_units[None]):

    """
    node wrapping lasagne's DenseLayer
    """

    def compute_output(self):
        in_var = self.get_input()
        num_units = self.find_hyperparameter("num_units")
        num_inputs = int(np.prod(in_var.shape[1:]))
        self.create_variable(
            "W",
            shape=(num_inputs, num_units),
            is_shared=True,
            tags=["parameter", "weight"]
        )
        self.create_variable(
            "b",
            shape=(num_units,),
            is_shared=True,
            tags=["parameter", "bias"]
        )
        self.input_layer = lasagne.layers.InputLayer(
            in_var.shape
        )
        self.dense_layer = lasagne.layers.DenseLayer(
            incoming=self.input_layer,
            num_units=num_units,
            W=self.W.variable,
            b=self.b.variable,
            nonlinearity=lasagne.nonlinearities.identity,
        )
        out_variable = self.dense_layer.get_output_for(in_var.variable)
        out_shape = self.dense_layer.get_output_shape_for(in_var.shape)
        self.create_variable(
            "result",
            variable=out_variable,
            shape=out_shape,
        )
        return dict(
            default=self.result,
        )


class ReLUNode(Node, Fields.name):

    """
    rectified linear unit
    """

    def compute_output(self):
        input = self.get_input()
        in_variable = input.variable
        out_variable = lasagne.nonlinearities.rectify(in_variable)
        self.create_variable(
            "result",
            variable=out_variable,
            shape=input.shape,
        )
        return dict(
            default=self.result,
        )


class SGDNode(WrapperNode,
              Fields.name.node.cost_reference[None].learning_rate[None]):

    """
    node that provides updates via SGD
    """

    def architecture_children(self):
        return [self.node]

    def init_state(self):
        cost_reference = self.find_hyperparameter("cost_reference")
        self.graph.add_dependency(cost_reference,
                                  self.name,
                                  to_key="cost")
        self.forward_input_to(self.node.name)
        self.take_input_from(self.node.name)

    def compute_update_deltas(self):
        cost = self.get_input(to_key="cost").variable
        parameters = self.find_variables_in_subtree(["parameter"])
        learning_rate = self.find_hyperparameter("learning_rate")
        updates = lasagne.updates.sgd(cost,
                                      [parameter.variable
                                       for parameter in parameters],
                                      learning_rate)
        return UpdateDeltas.from_updates(updates)
