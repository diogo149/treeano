"""
performs inverse operation of a single node by applying the
its partial derivative respect to its input
"""

import treeano
import theano
import theano.tensor as T

fX = theano.config.floatX


@treeano.register_node("inverse")
class InverseNode(treeano.NodeImpl):

    hyperparameter_names = ("reference",)

    def compute_output(self, network, in_vw):
        reference_name = network.find_hyperparameter(["reference"])
        ref_network = network[reference_name]

        in_var = in_vw.variable
        reference_input_vw = ref_network.get_input_vw("default")
        reference_input = reference_input_vw.variable
        reference_output_vw = ref_network.get_vw("default")
        reference_output = reference_output_vw.variable

        out_var = T.grad(None,
                         wrt=reference_input,
                         known_grads={reference_output: in_var})

        network.create_vw(
            'default',
            variable=out_var,
            shape=reference_input_vw.shape,
            tags={'output'}
        )
