import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


@treeano.register_node("concatenate_negation")
class ConcatenateNegationNode(treeano.NodeImpl):

    """
    concatenates a negated copy of the actviations on a specified axis
    """

    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], 1)

        in_var = in_vw.variable
        out_var = T.concatenate([in_var, -in_var], axis=axis)

        out_shape = list(in_vw.shape)
        if out_shape[axis] is not None:
            out_shape[axis] *= 2
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
