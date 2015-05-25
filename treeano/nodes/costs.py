"""
nodes for costs/losses
"""

import theano.tensor as T

from .. import core
from . import containers

AGGREGATORS = {
    'mean': T.mean,
    'sum': T.sum,
}


class AggregatorNode(core.NodeImpl):

    """
    node that aggregates a tensor into a scalar
    """

    hyperparameter_names = ("aggregator",)

    def compute_output(self, network, in_var):
        aggregator = network.find_hyperparameter(["aggregator"], "mean")
        aggregator_fn = AGGREGATORS.get(aggregator, aggregator)
        network.create_variable(
            "default",
            variable=aggregator_fn(in_var.variable),
            shape=(),
            tags={"output"}
        )


@core.register_node("elementwise_prediction_cost")
class ElementwisePredictionCostNode(core.WrapperNodeImpl):

    """
    node for computing the element-wise cost of predictions given a target
    """

    children_container = core.DictChildrenContainerSchema(
        preds=core.ChildContainer,
        target=core.ChildContainer,
    )
    hyperparameter_names = ("loss_function",)
    input_keys = ("preds_output", "target_output")

    def init_state(self, network):
        """
        by default, forward input to both preds and target
        """
        for child_name in ["preds", "target"]:
            node = self._children[child_name].children
            # forward input
            network.forward_input_to(node.name)
            # take output
            network.take_output_from(node.name,
                                     to_key="%s_output" % child_name)

    def compute_output(self, network, preds, target):
        loss_function = network.find_hyperparameter(["loss_function"])
        network.create_variable(
            "default",
            variable=loss_function(preds.variable, target.variable),
            shape=preds.shape,
            tags={"output"}
        )


@core.register_node("prediction_cost")
class PredictionCostNode(core.WrapperNodeImpl):

    """
    node for computing the cost of predictions given a target

    applies an element-wise cost, then aggregates together the costs
    """

    children_container = ElementwisePredictionCostNode.children_container
    hyperparameter_names = (ElementwisePredictionCostNode.hyperparameter_names
                            + AggregatorNode.hyperparameter_names)

    def architecture_children(self):
        return [containers.SequentialNode(
            self.name + "_sequential",
            [ElementwisePredictionCostNode(
                self.name + "_elementwise",
                self._children.children),
             AggregatorNode(self.name + "_aggregator")])]
