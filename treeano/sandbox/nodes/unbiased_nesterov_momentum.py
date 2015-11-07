import treeano


@treeano.register_node("unbiased_nesterov_momentum")
class UnbiasedNesterovMomentumNode(treeano.Wrapper1NodeImpl):

    """
    similar to NesterovMomentumNode, but includes a term to unbias the
    momentum update (similar to adam's unbias term)
    """

    # TODO add way to filter parameters and only apply to a subset
    hyperparameter_names = ("momentum",)

    def mutate_update_deltas(self, network, update_deltas):
        momentum = network.find_hyperparameter(["momentum"], 0.9)
        shared_vws = network.find_vws_in_subtree(is_shared=True)

        # keep count state only once
        t_vw = network.create_vw(
            "nesterov_momentum_count",
            shape=(),
            is_shared=True,
            tags={"state"},
            default_inits=[],
        )
        t = t_vw.variable
        new_t = t + 1
        update_deltas[t] = new_t - t
        # NOTE: assumes constant momentum
        unbias_factor = (1 - momentum) / (1 - momentum ** (new_t + 1))

        for vw in shared_vws:
            var = vw.variable
            if var in update_deltas:
                velocity_vw = network.create_vw(
                    "nesterov_momentum_velocity(%s)" % vw.name,
                    shape=vw.shape,
                    is_shared=True,
                    tags={"state"},
                    default_inits=[],
                )
                velocity = velocity_vw.variable
                delta = update_deltas[var]
                new_velocity = momentum * velocity + delta
                update_deltas[velocity] = new_velocity - velocity
                update_deltas[var] \
                    = (delta + momentum * new_velocity) * unbias_factor
