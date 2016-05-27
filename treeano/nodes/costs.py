"""
nodes for costs/losses
"""

import theano.tensor as T

from .. import core
from .. import utils
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

    def compute_output(self, network, in_vw):
        aggregator = network.find_hyperparameter(["aggregator"], "mean")
        aggregator_fn = AGGREGATORS.get(aggregator, aggregator)
        network.create_vw(
            "default",
            variable=aggregator_fn(in_vw.variable),
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
        weight=core.ChildContainer,
    )
    hyperparameter_names = ("cost_function",)
    input_keys = ("pred_output", "target_output", "weight_output")

    def architecture_children(self):
        children = self.raw_children()
        pred_node = children["pred"]
        target_node = children["target"]

        if "weight" in children:
            weight_node = children["weight"]
        else:
            weight_node = simple.ConstantNode(self.name + "_weight",
                                              value=1)
        return [pred_node, target_node, weight_node]

    def init_state(self, network):
        """
        by default, forward input to both pred and target
        """
        for child_name, node in zip(["pred", "target", "weight"],
                                    self.architecture_children()):
            # forward input
            network.forward_input_to(node.name)
            # take output
            network.take_output_from(node.name,
                                     to_key="%s_output" % child_name)

    def compute_output(self, network, pred, target, weight):
        cost_function = network.find_hyperparameter(["cost_function"])
        out_var = weight.variable * cost_function(pred.variable,
                                                  target.variable)
        network.create_vw(
            "default",
            variable=out_var,
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
                self.raw_children()),
             AggregatorNode(self.name + "_aggregator")])]

    def compute_output(self, network, in_vw):
        # output the children's output
        super(TotalCostNode, self).compute_output(network, in_vw)
        if network.find_hyperparameter(["monitor"]):
            # add monitoring
            network.create_vw(
                "cost",
                variable=in_vw.variable,
                shape=(),
                tags={"monitor"},
            )


@core.register_node("auxiliary_cost")
class AuxiliaryCostNode(core.WrapperNodeImpl):

    children_container = core.DictChildrenContainerSchema(
        target=core.ChildContainer,
        pre_cost=core.ChildContainer,
        weight=core.ChildContainer,
    )
    hyperparameter_names = ("cost_reference",
                            "cost_function",
                            "cost_weight")

    def architecture_children(self):
        children = self.raw_children()
        target = children["target"]
        nodes = []

        # allow for adding auxiliar nodes between input and cost
        if "pre_cost" in children:
            nodes.append(children["pre_cost"])

        cost_children = {"pred": simple.IdentityNode(self.name + "_identity"),
                         "target": target}
        if "weight" in children:
            cost_children["weight"] = children["weight"]
        # TODO parameterize cost node (to one that may not just be a function)
        # ie. something stateful
        nodes.append(TotalCostNode(self.name + "_cost", cost_children))
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


@core.register_node("l2_penalty")
class L2PenaltyNode(core.Wrapper1NodeImpl):

    """
    applies L2 penalty on weights
    """

    hyperparameter_names = ("l2_weight",
                            "cost_reference",
                            "to_key")

    def init_long_range_dependencies(self, network):
        network.forward_output_to(
            network.find_hyperparameter(["cost_reference"]),
            from_key="l2_cost",
            to_key=network.find_hyperparameter(["to_key"],
                                               "l2"))

    def compute_output(self, network, *args):
        # have output pass through
        super(L2PenaltyNode, self).compute_output(network, *args)

        l2_weight = network.find_hyperparameter(["l2_weight"])

        weight_vws = network.find_vws_in_subtree(tags=["weight"])
        weights = [vw.variable for vw in weight_vws]

        l2_penalty_var = utils.smart_sum([T.sum(w ** 2) for w in weights])

        network.create_vw(
            name="l2_cost",
            variable=l2_penalty_var * l2_weight,
            shape=(),
            tags={"monitor"},
        )
