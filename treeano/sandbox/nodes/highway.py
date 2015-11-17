"""
from "Highway Networks"
http://arxiv.org/abs/1505.00387
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("highway")
class HighwayNode(treeano.WrapperNodeImpl):

    """
    takes in a transform node and a gate node (that return variables with
    the same shape as their input) and combines them together in a highway
    network

    g = gate(input)
    result = g * transform(input) + (1 - g) * input
    """

    children_container = treeano.core.DictChildrenContainerSchema(
        transform=treeano.core.ChildContainer,
        gate=treeano.core.ChildContainer)

    hyperparameter_names = ()  # TODO add initial bias parameter

    def architecture_children(self):
        children = self.raw_children()
        gate = children["gate"]
        transform = children["transform"]

        # prepare gates
        transform_gate = tn.SequentialNode(
            self.name + "_transformgate",
            [gate,
             # add initial value as bias instead
             # TODO parameterize
             tn.AddConstantNode(self.name + "_biastranslation", value=-4),
             tn.SigmoidNode(self.name + "_transformgatesigmoid")])
        # carry gate = 1 - transform gate
        carry_gate = tn.SequentialNode(
            self.name + "_carrygate",
            [tn.ReferenceNode(self.name + "_transformgateref",
                              reference=transform_gate.name),
             tn.MultiplyConstantNode(self.name + "_invert", value=-1),
             tn.AddConstantNode(self.name + "_add", value=1)])

        # combine with gates
        gated_transform = tn.ElementwiseProductNode(
            self.name + "_gatedtransform",
            [transform_gate, transform])
        gated_carry = tn.ElementwiseProductNode(
            self.name + "_gatedcarry",
            [carry_gate, tn.IdentityNode(self.name + "_carry")])
        res = tn.ElementwiseSumNode(
            self.name + "_res",
            [gated_carry, gated_transform])
        return [res]


def HighwayDenseNode(name, nonlinearity_node, **hyperparameters):
    return tn.HyperparameterNode(
        name,
        HighwayNode(
            name + "_highway",
            {"transform": tn.SequentialNode(
                name + "_transform",
                [tn.DenseNode(name + "_transformdense"),
                 nonlinearity_node]),
             "gate": tn.DenseNode(name + "_gatedense")}),
        **hyperparameters)


def HighwayDnnConv2DNode(name, nonlinearity_node, **hyperparameters):
    return tn.HyperparameterNode(
        name,
        HighwayNode(
            name + "_highway",
            {"transform": tn.SequentialNode(
                name + "_transform",
                [tn.DnnConv2DNode(name + "_transformconv"),
                 nonlinearity_node]),
             "gate": tn.DnnConv2DNode(name + "_gateconv")}),
        **hyperparameters)
