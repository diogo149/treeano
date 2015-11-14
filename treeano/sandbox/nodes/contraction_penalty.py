"""
from "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction"

NOTES:
- implementation are slow
- doesn't seem to play nicely with ReLU or dropout
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("elementwise_contraction_penalty")
class ElementwiseContractionPenaltyNode(treeano.NodeImpl):

    """
    calculates an elementwise contraction penalty between the input of the node
    and an input_reference (what the input of the input of the node is)

    example use case for elementwise computation: weighting each input
    differently

    NOTE: more efficient implementations exist for affine maps followed by
    elementwise nonlinearities
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/autoencoder.py
    https://github.com/caglar/autoencoders/blob/master/ca.py
    """

    hyperparameter_names = ("input_reference",)
    input_keys = ("default", "transform_input")

    def init_long_range_dependencies(self, network):
        reference = network.find_hyperparameter(["input_reference"])
        network.take_output_from(reference,
                                 to_key="transform_input")

    def compute_output(self, network, h_vw, x_vw):
        batch_axis = network.find_hyperparameter(["batch_axis"])
        if batch_axis is None:
            # NOTE: this code path is not tested!
            jacobian = T.jacobian(h_vw.variable.ravel(), x_vw.variable)
            res = (jacobian ** 2).mean()
            res_shape = ()
        else:
            batch_size = h_vw.symbolic_shape()[batch_axis]
            # sum across batch to avoid disconnected input error
            # ravel to be a vector
            h_var = h_vw.variable.sum(axis=batch_axis).ravel()
            x_var = x_vw.variable
            # shape of result = h_var.shape + x_var.shape
            jacobian = T.jacobian(h_var, x_var)
            # put batch axis as first dimension
            # adding 1 to batch axis, because len(h_var.shape) == 1
            swapped_jacobian = jacobian.swapaxes(0, batch_axis + 1)
            # convert to a matrix and mean over elements in a batch
            reshaped_jacobian = swapped_jacobian.reshape((batch_size, -1))
            res = (reshaped_jacobian ** 2).mean(axis=1)
            res_shape = (h_vw.shape[batch_axis],)
        network.create_vw(
            "default",
            variable=res,
            shape=res_shape,
            tags={"output"},
        )


@treeano.register_node("auxiliary_contraction_penalty")
class AuxiliaryContractionPenaltyNode(treeano.Wrapper1NodeImpl):

    """
    returns the output of the inner node, and passes the computed penalty
    to the send_to node
    """

    hyperparameter_names = ("cost_reference",
                            "cost_weight")

    def architecture_children(self):
        inner = self.raw_children()
        input_node = tn.IdentityNode(self.name + "_input")
        return [
            tn.SequentialNode(
                self.name + "_sequential",
                [input_node,
                 inner,
                 tn.AuxiliaryNode(
                     self.name + "_auxiliary",
                     tn.SequentialNode(
                         self.name + "_innerseq",
                         [ElementwiseContractionPenaltyNode(
                             self.name + "_contractionpenalty",
                             input_reference=input_node.name),
                          tn.AggregatorNode(self.name + "_aggregator"),
                          tn.MultiplyConstantNode(
                              self.name + "_multiplyweight"),
                          tn.SendToNode(self.name + "_sendto",
                                        to_key=self.name)]))])]

    def init_long_range_dependencies(self, network):
        network.forward_hyperparameter(self.name + "_sendto",
                                       "send_to_reference",
                                       ["cost_reference"])

    def init_state(self, network):
        super(AuxiliaryContractionPenaltyNode, self).init_state(network)
        network.forward_hyperparameter(self.name + "_multiplyweight",
                                       "value",
                                       ["cost_weight"],
                                       1)
