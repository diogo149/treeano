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


@treeano.register_node("irregular_length_max")
class IrregularLengthMaxNode(treeano.NodeImpl):

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
            variable=in_vw.variable.max(axis=1),
            # drop axis 1
            shape=in_vw.shape[0:1] + in_vw.shape[2:],
            tags={"output"},
        )
        
