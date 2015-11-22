import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from theano.sandbox.rng_mrg import MRG_RandomStreams


@treeano.register_node("randomized_relu")
class RandomizedReLUNode(treeano.NodeImpl):

    """
    from "Empirical Evaluation of Rectified Activations in Convolutional
    Network"
    http://arxiv.org/abs/1505.00853
    """
    hyperparameter_names = ("alpha_lower",
                            "alpha_upper",
                            "deterministic")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        deterministic = network.find_hyperparameter(["deterministic"])
        l = network.find_hyperparameter(["alpha_lower"],
                                        3)
        u = network.find_hyperparameter(["alpha_upper"],
                                        8)

        if deterministic:
            negative_coefficient = 2.0 / (l + u)
        else:
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            alphas = srng.uniform(size=in_vw.symbolic_shape(),
                                  low=l,
                                  high=u)
            negative_coefficient = 1.0 / alphas

        # return output
        network.create_vw(
            "default",
            variable=treeano.utils.rectify(
                in_vw.variable,
                negative_coefficient=negative_coefficient),
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("uniform_randomized_relu")
class UniformRandomizedReLUNode(treeano.NodeImpl):

    """
    like RandomizedReLUNode, but instead of sampling from 1 / uniform(l, u),
    sample from uniform(l, u)
    """

    hyperparameter_names = ("alpha_lower",
                            "alpha_upper",
                            "deterministic")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        deterministic = network.find_hyperparameter(["deterministic"])
        l = network.find_hyperparameter(["alpha_lower"],
                                        1 / 8.)
        u = network.find_hyperparameter(["alpha_upper"],
                                        1 / 3.)

        if deterministic:
            negative_coefficient = (l + u) / 2.
        else:
            # TODO save this state so that we can seed the rng
            srng = MRG_RandomStreams()
            negative_coefficient = srng.uniform(size=in_vw.symbolic_shape(),
                                                low=l,
                                                high=u)

        # return output
        network.create_vw(
            "default",
            variable=treeano.utils.rectify(
                in_vw.variable,
                negative_coefficient=negative_coefficient),
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("random_walk_relu")
class RandomWalkReLUNode(treeano.NodeImpl):

    """
    leaky ReLU node, where leak alpha changes randomly over time
    """

    hyperparameter_names = ("step_size",
                            "initial_alpha",
                            "inits")

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        initial_alpha = network.find_hyperparameter(
            ["initial_alpha"],
            0)
        alpha = network.create_vw(
            "alpha",
            is_shared=True,
            shape=(in_vw.shape[1],),
            tags={"state"},
            default_inits=[treeano.inits.ConstantInit(initial_alpha)],
        ).variable

        pattern = ["x"] * in_vw.ndim
        pattern[1] = 0
        alpha_b = alpha.dimshuffle(*pattern)

        # return output
        network.create_vw(
            "default",
            variable=treeano.utils.rectify(in_vw.variable,
                                           negative_coefficient=alpha_b),
            shape=in_vw.shape,
            tags={"output"},
        )

    def new_update_deltas(self, network):
        alpha_vw = network.get_vw("alpha")
        step_size = network.find_hyperparameter(["step_size"])
        # NOTE: each MRG_RandomStreams has the same seed, so
        # all nodes with the same shape end up with the same alphas
        srng = MRG_RandomStreams()
        steps = srng.uniform(size=alpha_vw.shape,
                             low=-step_size,
                             high=step_size)
        # TODO clip value of alpha (to prevent it becoming linear)
        return treeano.UpdateDeltas({alpha_vw.variable: steps})
