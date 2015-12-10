"""
scaling updates by initial standard deviation similar to techniques from
http://arxiv.org/abs/1511.00363

NOTE: mostly copy-pasted from treeano.nodes.updates - changed lines are
marked with a HACK
"""


import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("scaled_adam")
class ScaledAdamNode(treeano.nodes.updates.StandardUpdatesNode):

    """
    node that provides updates via Adam update rule
    based on Adam update rule v7 (http://arxiv.org/abs/1412.6980)
    """

    hyperparameter_names = ("adam_learning_rate",
                            "adam_alpha",
                            "learning_rate",
                            "adam_beta1",
                            "beta1",
                            "adam_beta2",
                            "beta2",
                            "adam_epsilon",
                            "epsilon",
                            "scale_function",)

    def _new_update_deltas(self, network, parameter_vws, grads):
        # alpha / stepsize / learning rate are all the same thing
        # using alpha because that is what is used in the paper
        alpha = network.find_hyperparameter(["adam_learning_rate",
                                             "adam_alpha",
                                             "learning_rate"],
                                            0.001)
        beta1 = network.find_hyperparameter(["adam_beta1",
                                             "beta1"],
                                            0.9)
        beta2 = network.find_hyperparameter(["adam_beta2",
                                             "beta2"],
                                            0.999)
        epsilon = network.find_hyperparameter(["adam_epsilon",
                                               "epsilon"],
                                              1e-8)
        # HACK part 1: different from adam
        scale_fn = network.find_hyperparameter(["scale_function"],
                                               treeano.utils.identity)

        update_deltas = treeano.UpdateDeltas()

        # keep count state only once
        t_vw = network.create_vw(
            "adam_count",
            shape=(),
            is_shared=True,
            tags={"state"},
            default_inits=[],
        )
        t = t_vw.variable
        new_t = t + 1
        update_deltas[t] = new_t - t

        # compute some values only once
        # unbias terms to take into account initializing with 0
        # NOTE: unbias terms assume constant beta1/beta2
        m_unbias_term = 1 - beta1 ** new_t
        v_unbias_term = T.sqrt(1 - beta2 ** new_t)
        epsilon_hat = epsilon * v_unbias_term
        alpha_t = alpha * v_unbias_term / m_unbias_term

        for parameter_vw, grad in zip(parameter_vws, grads):
            # biased 1st moment estimate
            # moving average of gradient
            m_vw = network.create_vw(
                "adam_m(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )
            # 2nd moment
            # moving average of squared gradient
            v_vw = network.create_vw(
                "adam_v(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )

            m = m_vw.variable
            v = v_vw.variable

            # new value for 1st moment estimate
            new_m = beta1 * m + (1 - beta1) * grad
            # new value for 2nd moment estimate
            new_v = beta2 * v + (1 - beta2) * T.sqr(grad)

            parameter_delta = - alpha_t * new_m / (T.sqrt(new_v) + epsilon_hat)

            # HACK  part 2: different from standard adam
            initial_std = treeano.utils.as_fX(np.std(parameter_vw.value))
            # prevent multiplying by 0 std
            if initial_std > 0:
                parameter_delta *= scale_fn(initial_std)

            update_deltas[m] = new_m - m
            update_deltas[v] = new_v - v
            update_deltas[parameter_vw.variable] = parameter_delta

        return update_deltas


@treeano.register_node("scaled_sgd")
class ScaledSGDNode(treeano.nodes.updates.StandardUpdatesNode):

    """
    node that provides updates via SGD
    """

    hyperparameter_names = ("sgd_learning_rate",
                            "learning_rate")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["sgd_learning_rate",
                                                     "learning_rate"],
                                                    0.1)
        # HACK changes the rest of this node... mostly restructuring
        deltas = {}
        for vw, grad in zip(parameter_vws, grads):
            initial_std = np.std(vw.value)
            # prevent multiplying by 0 std
            if initial_std == 0:
                initial_std = 1.0
            factor = treeano.utils.as_fX(-learning_rate * initial_std ** 2)
            deltas[vw.variable] = factor * grad
        return treeano.UpdateDeltas(deltas)


@treeano.register_node("scaled_nesterovs_accelerated_gradient")
class ScaledNesterovsAcceleratedGradientNode(treeano.WrapperNodeImpl):

    """
    node that provides updates via SGD Nesterov's Accelerated Gradient Descent
    """

    children_container = treeano.core.DictChildrenContainerSchema(
        cost=treeano.core.ChildContainer,
        subtree=treeano.core.ChildContainer,
    )

    hyperparameter_names = (ScaledSGDNode.hyperparameter_names  # HACK
                            + tn.NesterovMomentumNode.hyperparameter_names)

    def architecture_children(self):
        children = self.raw_children()
        momentum_node = tn.NesterovMomentumNode(self.name + "_momentum",
                                                children["subtree"])
        new_children = {"subtree": momentum_node,
                        "cost": children["cost"]}
        # HACK
        return [ScaledSGDNode(self.name + "_sgd", new_children)]
