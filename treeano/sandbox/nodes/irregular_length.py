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
