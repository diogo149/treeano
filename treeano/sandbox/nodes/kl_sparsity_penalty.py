"""
for applying a sparsity (for saturating nonlinearities) using KL-divergence
of a bernoulli distribution

unsure of origin, but see the following pdf for info:
http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


def _bernoulli_kl_divergence(p, outputs):
    """
    p:
    sparsity parameter (target sparsity)

    outputs:
    actual network outputs
    """
    return (p * T.log(p)
            - p * T.log(outputs)
            + (1 - p) * T.log(1 - p)
            - (1 - p) * T.log(1 - outputs))


@treeano.register_node("elementwise_kl_sparsity_penalty")
class ElementwiseKLSparsityPenaltyNode(treeano.NodeImpl):

    hyperparameter_names = ("target_sparsity",
                            "sparsity",
                            "min_value",
                            "max_value")

    def compute_output(self, network, in_vw):
        p = network.find_hyperparameter(["target_sparsity",
                                         "sparsity"])
        min_value = network.find_hyperparameter(["min_value"], 0)
        max_value = network.find_hyperparameter(["max_value"], 1)
        scaled_output = (in_vw.variable - min_value) / (max_value - min_value)
        cost = _bernoulli_kl_divergence(p, scaled_output)
        network.create_vw(
            "default",
            variable=cost,
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("auxiliary_kl_sparsity_penalty")
class AuxiliaryKLSparsityPenaltyNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = (
        ("cost_reference",
         "cost_weight")
        + ElementwiseKLSparsityPenaltyNode.hyperparameter_names)

    def architecture_children(self):
        return [
            tn.AuxiliaryNode(
                self.name + "_auxiliary",
                tn.SequentialNode(
                    self.name + "_sequential",
                    [ElementwiseKLSparsityPenaltyNode(
                        self.name + "_sparsitypenalty"),
                     tn.AggregatorNode(self.name + "_aggregator"),
                     tn.MultiplyConstantNode(self.name + "_multiplyweight"),
                     tn.SendToNode(self.name + "_sendto", to_key=self.name)]))]

    def init_long_range_dependencies(self, network):
        network.forward_hyperparameter(self.name + "_sendto",
                                       "send_to_reference",
                                       ["cost_reference"])

    def init_state(self, network):
        super(AuxiliaryKLSparsityPenaltyNode, self).init_state(network)
        network.forward_hyperparameter(self.name + "_multiplyweight",
                                       "value",
                                       ["cost_weight"],
                                       1)
