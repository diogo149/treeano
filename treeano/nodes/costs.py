"""
nodes for costs/losses
"""

import theano.tensor as T

from .. import core
from . import simple
from . import containers

AGGREGATORS = {
    'mean': T.mean,
    'sum': T.sum,
}


@core.register_node("aggregator")
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


@core.register_node("elementwise_cost")
class ElementwiseCostNode(core.WrapperNodeImpl):

    """
    node for computing the element-wise cost of predictions given a target
    """

    children_container = core.DictChildrenContainerSchema(
        pred=core.ChildContainer,
        target=core.ChildContainer,
    )
    hyperparameter_names = ("cost_function",)
    input_keys = ("pred_output", "target_output")

    def init_state(self, network):
        """
        by default, forward input to both pred and target
        """
        for child_name in ["pred", "target"]:
            node = self._children[child_name].children
            # forward input
            network.forward_input_to(node.name)
            # take output
            network.take_output_from(node.name,
                                     to_key="%s_output" % child_name)

    def compute_output(self, network, pred, target):
        cost_function = network.find_hyperparameter(["cost_function"])
        network.create_variable(
            "default",
            variable=cost_function(pred.variable, target.variable),
            shape=pred.shape,
            tags={"output"}
        )


@core.register_node("total_cost")
class TotalCostNode(core.WrapperNodeImpl):

    """
    node for computing the cost of predictions given a target

    applies an element-wise cost, then aggregates together the costs
    """

    children_container = ElementwiseCostNode.children_container
    hyperparameter_names = (ElementwiseCostNode.hyperparameter_names
                            + AggregatorNode.hyperparameter_names)

    def architecture_children(self):
        return [containers.SequentialNode(
            self.name + "_sequential",
            [ElementwiseCostNode(
                self.name + "_elementwise",
                self._children.children),
             AggregatorNode(self.name + "_aggregator")])]


@core.register_node("auxiliary_cost")
class AuxiliaryCostNode(core.WrapperNodeImpl):

    children_container = core.DictChildrenContainerSchema(
        target=core.ChildContainer,
        pre_cost=core.ChildContainer,
    )
    hyperparameter_names = ("cost_reference",
                            "cost_function",
                            "cost_weight")

    def architecture_children(self):
        target = self._children["target"].children
        nodes = []

        # allow for adding auxiliar nodes between input and cost
        pre_cost_container = self._children.get("pre_cost")
        if pre_cost_container is not None:
            pre_cost_node = pre_cost_container.children
            nodes.append(pre_cost_node)

        # TODO parameterize cost node (to one that may not just be a function)
        # ie. something stateful
        nodes.append(TotalCostNode(
            self.name + "_cost",
            {"pred": simple.IdentityNode(self.name + "_identity"),
             "target": target}))
        nodes += [
            simple.MultiplyConstantNode(
                self.name + "_multiplyweight"),
            simple.SendToNode(self.name + "_sendto",
                              to_key=self.name)
        ]

        return [
            containers.AuxiliaryNode(
                self.name + "_auxiliary",
                containers.SequentialNode(
                    self.name + "_sequential",
                    nodes))]

    def init_long_range_dependencies(self, network):
        # must be set in init_long_range_dependencies, because long range
        # dependencies depend on reference
        network.forward_hyperparameter(self.name + "_multiplyweight",
                                       "value",
                                       ["cost_weight"],
                                       1)
        network.forward_hyperparameter(self.name + "_sendto",
                                       "send_to_reference",
                                       ["cost_reference"])
