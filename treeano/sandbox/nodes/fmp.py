"""
from "Fractional Max-Pooling" http://arxiv.org/abs/1412.6071
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
import treeano.theano_extensions.fractional_max_pooling as fmp


@treeano.register_node("disjoint_pseudorandom_fractional_max_pool_2d")
class DisjointPseudorandomFractionalMaxPool2DNode(treeano.NodeImpl):

    hyperparameter_names = ("fmp_alpha",
                            "fmp_u")

    def compute_output(self, network, in_vw):
        assert in_vw.ndim == 4
        alpha = network.find_hyperparameter(["fmp_alpha"])
        u = network.find_hyperparameter(["fmp_u"])
        op = fmp.DisjointPseudorandomFractionalMaxPooling2DOp(
            alpha=alpha,
            u=u
        )
        out_shape = list(in_vw.shape)
        # currently must be the same height and width
        assert out_shape[2] == out_shape[3]
        if out_shape[2] is not None:
            out_shape[2] = out_shape[3] = op.output_length(out_shape[2])
        out_shape = tuple(out_shape)
        network.create_vw(
            "default",
            variable=op(in_vw.variable),
            shape=out_shape,
            tags={"output"},
        )


@treeano.register_node("overlapping_random_fractional_max_pool_2d")
class OverlappingRandomFractionalMaxPool2DNode(treeano.NodeImpl):

    """
    modified from https://github.com/Lasagne/Lasagne/pull/171/files
    """

    hyperparameter_names = ("pool_size",
                            "pool_function")

    def compute_output(self, network, in_vw):
        if network.find_hyperparameter(["deterministic"]):
            import warnings
            warnings.warn("OverlappingRandomFractionalMaxPool2DNode has "
                          "no deterministic implementation")

        pool_size = network.find_hyperparameter(["pool_size"])
        assert len(pool_size) == 2
        for alpha in pool_size:
            assert 1 < alpha < 2
        pool_fn = network.find_hyperparameter(["pool_function"], T.max)

        # NOTE: MRG_RandomStreams doesn't have "permutation"
        srng = T.shared_randomstreams.RandomStreams()

        def theano_shuffled(in_vw):
            n = in_vw.shape[0]

            shuffled = T.permute_row_elements(in_vw.T, srng.permutation(n=n)).T
            return shuffled

        out_shape = list(in_vw.shape)
        for axis, alpha in zip([2, 3], pool_size):
            out_shape[axis] = int(np.ceil(float(out_shape[axis]) / alpha))
        out_shape = tuple(out_shape)

        n_in0, n_in1 = in_vw.shape[2:]
        n_out0, n_out1 = out_shape[2:]

        # Variable stride across the input creates fractional reduction
        a = theano.shared(
            np.array([2] * (n_in0 - n_out0) + [1] * (2 * n_out0 - n_in0)))
        b = theano.shared(
            np.array([2] * (n_in1 - n_out1) + [1] * (2 * n_out1 - n_in1)))

        # Randomize the input strides
        a = theano_shuffled(a)
        b = theano_shuffled(b)

        # Convert to input positions, starting at 0
        a = T.concatenate(([0], a[:-1]))
        b = T.concatenate(([0], b[:-1]))
        a = T.cumsum(a)
        b = T.cumsum(b)

        # Positions of the other corners
        c = T.clip(a + 1, 0, n_in0 - 1)
        d = T.clip(b + 1, 0, n_in1 - 1)

        # Index the four positions in the pooling window and stack them
        in_var = in_vw.variable
        temp = T.stack(in_var[:, :, a, :][:, :, :, b],
                       in_var[:, :, c, :][:, :, :, b],
                       in_var[:, :, a, :][:, :, :, d],
                       in_var[:, :, c, :][:, :, :, d])

        out_var = pool_fn(temp, axis=0)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"}
        )
