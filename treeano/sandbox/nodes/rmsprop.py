import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("deepmind_rmsprop")
class DeepMindRMSPropNode(tn.StandardUpdatesNode):

    """
    see:
    - https://github.com/soumith/deepmind-atari/blob/master/dqn/NeuralQLearner.lua
    - https://groups.google.com/forum/#!topic/deep-q-learning/_RFrmUALBQo
    """

    hyperparameter_names = ("learning_rate",
                            "rho",
                            "deepmind_rmsprop_epsilon",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 1e-2)
        rho = network.find_hyperparameter(["rho"], 0.95)
        epsilon = network.find_hyperparameter(
            ["deepmind_rmsprop_epsilon", "epsilon"], 0.01)

        update_deltas = treeano.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # exponential moving average of gradients
            g_avg = network.create_vw(
                "deepmind_rmsprop_gradients(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients squared
            g2_avg = network.create_vw(
                "deepmind_rmsprop_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated gradients squared
            new_g_avg = rho * g_avg + (1 - rho) * grad
            new_g2_avg = rho * g2_avg + (1 - rho) * T.sqr(grad)

            # calculate update
            std = T.sqrt(new_g2_avg - T.sqr(new_g_avg) + epsilon)
            deltas = -learning_rate * grad / std

            update_deltas[g_avg] = new_g_avg - g_avg
            update_deltas[g2_avg] = new_g2_avg - g2_avg
            update_deltas[parameter_vw.variable] = deltas

        return update_deltas


@treeano.register_node("graves_rmsprop")
class GravesRMSPropNode(tn.StandardUpdatesNode):

    """
    from http://arxiv.org/pdf/1308.0850v5.pdf
    """

    hyperparameter_names = ("learning_rate",
                            "rho",
                            "momentum",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 1e-4)
        rho = network.find_hyperparameter(["rho"], 0.95)
        momentum = network.find_hyperparameter(["momentum"], 0.9)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-4)

        update_deltas = treeano.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # momentum term
            delta = network.create_vw(
                "graves_rmsprop_delta(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients
            g_avg = network.create_vw(
                "graves_rmsprop_gradients(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients squared
            g2_avg = network.create_vw(
                "graves_rmsprop_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated gradients squared
            new_g_avg = rho * g_avg + (1 - rho) * grad
            new_g2_avg = rho * g2_avg + (1 - rho) * T.sqr(grad)

            # calculate update
            std = T.sqrt(new_g2_avg - T.sqr(new_g_avg) + epsilon)
            new_delta = momentum * delta - learning_rate * grad / std

            update_deltas[g_avg] = new_g_avg - g_avg
            update_deltas[g2_avg] = new_g2_avg - g2_avg
            update_deltas[delta] = new_delta - delta
            update_deltas[parameter_vw.variable] = new_delta

        return update_deltas


@treeano.register_node("std_rmsprop_with_momentum")
class StdRMSPropWithMomentumNode(tn.StandardUpdatesNode):

    """
    based on deepmind rmsprop
    """

    hyperparameter_names = ("learning_rate",
                            "momentum",
                            "rho",
                            "std_rmsprop_epsilon",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 1e-2)
        momentum = network.find_hyperparameter(["momentum"], 0.9)
        rho = network.find_hyperparameter(["rho"], 0.95)
        epsilon = network.find_hyperparameter(
            ["std_rmsprop_epsilon", "epsilon"],
            1e-8)

        update_deltas = treeano.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # exponential moving average of gradients for numerator
            g_avg_numer = network.create_vw(
                "std_rmsprop_gradients_momentum(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients for denominator
            g_avg_denom = network.create_vw(
                "std_rmsprop_gradients(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients squared
            g2_avg = network.create_vw(
                "std_rmsprop_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated state
            new_g_avg_numer = momentum * g_avg_numer + (1 - momentum) * grad
            new_g_avg_denom = rho * g_avg_denom + (1 - rho) * grad
            new_g2_avg = rho * g2_avg + (1 - rho) * T.sqr(grad)

            # calculate update
            std = T.sqrt(new_g2_avg - T.sqr(new_g_avg_denom) + epsilon)
            deltas = -learning_rate * new_g_avg_numer / std

            update_deltas[g_avg_numer] = new_g_avg_numer - g_avg_numer
            update_deltas[g_avg_denom] = new_g_avg_denom - g_avg_denom
            update_deltas[g2_avg] = new_g2_avg - g2_avg
            update_deltas[parameter_vw.variable] = deltas

        return update_deltas
