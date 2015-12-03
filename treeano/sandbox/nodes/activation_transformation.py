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


class NegatedInit(treeano.inits.WeightInit):

    """
    init specifically for units after ConcatenateNegationNode

    takes in a different init, and initializes the first half of with that
    init, and the second half with the negated version of that tensor

    rationale: ConcatenateNegationNode + this init + ReLU will initialize
    the network to be linear
    """

    def __init__(self, init, axis=1):
        self.init = init
        self.axis = axis

    def initialize_value(self, vw):
        # temporary variable wrapper with fake shape
        tmp_vw_shape = list(vw.shape)
        if tmp_vw_shape[self.axis] % 2 != 0:
            # this weight is probably not after a ConcatenateNegationNode,
            # so instead revert to initial init
            return self.init.initialize_value(vw)
        tmp_vw_shape[self.axis] /= 2
        tmp_vw_shape = tuple(tmp_vw_shape)
        tmp_vw = treeano.VariableWrapper(
            "tmp",
            shape=tmp_vw_shape,
            is_shared=True,
            inits=[],
        )

        val = self.init.initialize_value(tmp_vw)

        return np.concatenate([val, -val], axis=self.axis)
