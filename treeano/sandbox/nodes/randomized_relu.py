"""
from "Empirical Evaluation of Rectified Activations in Convolutional Network"
http://arxiv.org/abs/1505.00853
"""
import toolz
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn
from theano.sandbox.rng_mrg import MRG_RandomStreams


@treeano.register_node("randomized_relu")
class RandomizedReLUNode(treeano.NodeImpl):

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
