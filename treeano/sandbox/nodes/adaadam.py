import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("adaadam")
class AdaAdamNode(tn.StandardUpdatesNode):

    """
    """

    hyperparameter_names = ("learning_rate",
                            "beta1",
                            "beta2",
                            "epsilon",
                            "half_life_batches",
                            "clipped_batches")

    def _new_update_deltas(self, network, parameter_vws, grads):
        # alpha / stepsize / learning rate are all the same thing
        # using alpha because that is what is used in the paper
        alpha = network.find_hyperparameter(["learning_rate"],
                                            0.001)
        beta1 = network.find_hyperparameter(["beta1"],
                                            0.9)
        beta2 = network.find_hyperparameter(["beta2"],
                                            0.999)
        epsilon = network.find_hyperparameter(["epsilon"],
                                              1e-8)
        constant_root = network.find_hyperparameter(["constant_root"], None)
        normalize_denominator = network.find_hyperparameter(
            ["normalize_denominator"],
            True)

        update_deltas = treeano.UpdateDeltas()

        # keep count state only once
        t_vw = network.create_vw(
            "adaadam_count",
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

        if constant_root is None:
            h = network.find_hyperparameter(["half_life_batches"])
            # heuristic: set as half_life_batches by default
            c = network.find_hyperparameter(["clipped_batches"], h)
            f = 2.0 ** (1. / h)
            w0 = 2.0 * (1 / f) ** c
            w_state = network.create_vw(
                "adaadam_w",
                shape=(),
                is_shared=True,
                tags={"state"},
                default_inits=[treeano.inits.ConstantInit(w0)],
            ).variable
            update_deltas[w_state] = w_state * f - w_state
            # TODO parameterize bounds
            w = T.clip(w_state, 2.0, 10000.0)
        else:
            w = constant_root

        for parameter_vw, grad in zip(parameter_vws, grads):
            # biased 1st moment estimate
            # moving average of gradient
            m_vw = network.create_vw(
                "adaadam_m(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            )
            # 2nd moment
            # moving average of squared gradient
            v_vw = network.create_vw(
                "adaadam_v(%s)" % parameter_vw.name,
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

            orig_denom = T.sqrt(new_v)
            denom = T.pow(new_v, 1. / w)
            # FIXME try w/ and w/o normalizer
            if normalize_denominator:
                denom_normalizer = ((orig_denom.sum() + 1e-8)
                                    / (denom.sum() + 1e-8))
            else:
                denom_normalizer = 1

            if 1:
                parameter_delta = - alpha_t * new_m / ((denom + epsilon_hat)
                                                       * denom_normalizer)
            else:
                parameter_delta = - alpha_t * new_m / (denom * denom_normalizer
                                                       + epsilon_hat)

            update_deltas[m] = new_m - m
            update_deltas[v] = new_v - v
            update_deltas[parameter_vw.variable] = parameter_delta

        return update_deltas
