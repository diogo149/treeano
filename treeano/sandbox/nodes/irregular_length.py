import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import treeano.theano_extensions.irregular_length as il


@treeano.register_node("ungroup_irregular_length_tensors")
class UngroupIrregularLengthTensorsNode(treeano.NodeImpl):

    hyperparameter_names = ("lengths_reference",)
    input_keys = ("default", "lengths")

    def init_long_range_dependencies(self, network):
        network.take_output_from(
            network.find_hyperparameter(["lengths_reference"]),
            to_key="lengths")

    def compute_output(self, network, in_vw, lengths_vw):
        network.create_vw(
            "default",
            variable=il.ungroup_irregular_length_tensors(in_vw.variable,
                                                         lengths_vw.variable),
            shape=(None, None) + in_vw.shape[1:],
            tags={"output"},
        )


@treeano.register_node("irregular_length_mean")
class IrregularLengthMeanNode(treeano.NodeImpl):

    """
    meant to take in input from UngroupIrregularLengthTensorsNode
    """

    hyperparameter_names = ("lengths_reference",)
    input_keys = ("default", "lengths")

    def init_long_range_dependencies(self, network):
        network.take_output_from(
            network.find_hyperparameter(["lengths_reference"]),
            to_key="lengths")

    def compute_output(self, network, in_vw, lengths_vw):
        lengths = lengths_vw.variable.dimshuffle([0] + ["x"] * (in_vw.ndim - 2))
        network.create_vw(
            "default",
            variable=in_vw.variable.sum(axis=1) / lengths,
            # drop axis 1
            shape=in_vw.shape[0:1] + in_vw.shape[2:],
            tags={"output"},
        )


@treeano.register_node("_irregular_length_attention_softmax")
class _IrregularLengthAttentionSoftmaxNode(treeano.NodeImpl):

    """
    performs a softmax over irregular length, but takes into account
    the padding created by ungrouping by removing the contribution
    of those 0's to the denominator of a softmax
    """

    hyperparameter_names = ("lengths_reference",)
    input_keys = ("default", "lengths")

    def init_long_range_dependencies(self, network):
        network.take_output_from(
            network.find_hyperparameter(["lengths_reference"]),
            to_key="lengths")

    def compute_output(self, network, in_vw, lengths_vw):
        x = in_vw.variable
        x_max = x.max(axis=1, keepdims=True)
        e_x = T.exp(x - x_max)
        e_x_max_inv = T.exp(-x_max)
        lengths = lengths_vw.variable
        max_len = lengths.max()
        num_zeros = max_len - lengths
        num_zeros = num_zeros.dimshuffle([0, "x", "x"])
        denom = e_x.sum(axis=1, keepdims=True) - num_zeros * e_x_max_inv
        out_var = e_x / denom
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"},
        )


def irregular_length_attention_node(name,
                                    lengths_reference,
                                    num_units,
                                    output_units=None):
    """
    NOTE: if output_units is not None, this should be the number
    of input units
    """
    value_branch = UngroupIrregularLengthTensorsNode(
        name + "_ungroup_values",
        lengths_reference=lengths_reference)

    fc2_units = 1 if output_units is None else output_units
    attention_nodes = [
        tn.DenseNode(name + "_fc1", num_units=num_units),
        tn.ScaledTanhNode(name + "_tanh"),
        tn.DenseNode(name + "_fc2", num_units=fc2_units),
        UngroupIrregularLengthTensorsNode(
            name + "_ungroup_attention",
            lengths_reference=lengths_reference),
        _IrregularLengthAttentionSoftmaxNode(
            name + "_softmax",
            lengths_reference=lengths_reference),
    ]
    if output_units is None:
        attention_nodes += [
            tn.AddBroadcastNode(name + "_bcast", axes=(2,)),
        ]

    attention_branch = tn.SequentialNode(
        name + "_attention",
        attention_nodes)

    return tn.SequentialNode(
        name,
        [tn.ElementwiseProductNode(
            name + "_prod",
            [value_branch,
             attention_branch]),
         tn.SumNode(name + "_sum", axis=1)])
