import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("rms_normalization")
class RMSNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("epsilon",)

    def compute_output(self, network, in_vw):
        x = in_vw.variable
        epsilon = 1e-5

        kwargs = dict(axis=[dim for dim in range(x.ndim)
                            if dim != 0],
                      keepdims=True)
        gamma = network.create_vw(
            name="gamma",
            is_shared=True,
            shape=(in_vw.shape[1],),
            tags={"parameter"},
            default_inits=[],
        ).variable.dimshuffle("x", 0, *(["x"] * (in_vw.ndim - 2)))
        z = x * (T.exp(gamma) / T.sqrt(T.sqr(x).mean(**kwargs) + epsilon))
        network.create_vw(
            name="default",
            variable=z,
            shape=in_vw.shape,
            tags={"output"},
        )
