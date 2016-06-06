import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("monitored_adam")
class MonitoredAdamNode(tn.StandardUpdatesNode):

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
                            "epsilon")

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

            adam_term = new_m / (T.sqrt(new_v) + epsilon_hat)
            parameter_delta = - alpha_t * adam_term

            update_deltas[m] = new_m - m
            update_deltas[v] = new_v - v
            update_deltas[parameter_vw.variable] = parameter_delta

            network.create_vw(
                name="abs_mean(%s)" % parameter_vw.name,
                variable=abs(adam_term).mean(),
                shape=(),
                tags={"monitor"},
            )

        return update_deltas
