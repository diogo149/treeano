import numpy as np
import lasagne

from .. import core


@core.register_node("lasagne_dense")
class DenseNode(core.NodeImpl):

    """
    node wrapping lasagne's DenseLayer
    """

    hyperparameter_names = ("dense_num_units", "num_units")

    def compute_output(self, network, in_var):
        num_units = network.find_hyperparameter(
            ["dense_num_units", "num_units"])
        num_inputs = int(np.prod(in_var.shape[1:]))
        W = network.create_variable(
            "W",
            shape=(num_inputs, num_units),
            is_shared=True,
            tags=["parameter", "weight"]
        )
        b = network.create_variable(
            "b",
            shape=(num_units,),
            is_shared=True,
            tags=["parameter", "bias"]
        )
        input_layer = lasagne.layers.InputLayer(
            in_var.shape
        )
        dense_layer = lasagne.layers.DenseLayer(
            incoming=input_layer,
            num_units=num_units,
            W=W.variable,
            b=b.variable,
            nonlinearity=lasagne.nonlinearities.identity,
        )
        out_variable = dense_layer.get_output_for(in_var.variable)
        out_shape = dense_layer.get_output_shape_for(in_var.shape)
        network.create_variable(
            "default",
            variable=out_variable,
            shape=out_shape,
            tags={"output"}
        )


@core.register_node("lasagne_relu")
class ReLUNode(core.NodeImpl):

    """
    rectified linear unit
    """

    def compute_output(self, network, in_var):
        network.create_variable(
            "default",
            variable=lasagne.nonlinearities.rectify(in_var.variable),
            shape=in_var.shape,
            tags={"output"},
        )


@core.register_node("lasagne_sgd")
class SGDNode(core.Wrapper1NodeImpl):

    """
    node that provides updates via SGD
    """

    hyperparameter_names = ("cost_reference",
                            "reference",
                            "sgd_learning_rate",
                            "learning_rate")

    def new_update_deltas(self, network):
        cost_reference = network.find_hyperparameter(["cost_reference",
                                                      "reference"])
        cost = network[cost_reference].get_variable("default").variable
        parameters = network.find_variables_in_subtree(["parameter"])
        learning_rate = network.find_hyperparameter(["sgd_learning_rate",
                                                     "learning_rate"])
        updates = lasagne.updates.sgd(cost,
                                      [parameter.variable
                                       for parameter in parameters],
                                      learning_rate)
        return core.UpdateDeltas.from_updates(updates)
