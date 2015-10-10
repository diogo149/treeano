"""
nodes that provide updates for shared variables
"""
import abc

import toolz
import six
import numpy as np
import theano
import theano.tensor as T

from .. import core


@core.register_node("update_scale")
class UpdateScaleNode(core.Wrapper1NodeImpl):

    """
    scales updates from above the tree by multiplying by a constant scale
    factor
    """

    hyperparameter_names = ("update_scale_factor", "scale_factor")

    def mutate_update_deltas(self, network, update_deltas):
        # TODO parameterize which nodes to search for (eg. maybe we want
        # to scale state updates)
        scale_factor = network.find_hyperparameter(["update_scale_factor",
                                                    "scale_factor"])
        parameters = network.find_vws_in_subtree(tags=["parameter"])
        for parameter in parameters:
            update_deltas[parameter.variable] *= scale_factor


# ############################ standard updaters ############################
# ---
# updaters that take in the parameters and their gradient w.r.t. a cost


class StandardUpdatesNode(six.with_metaclass(abc.ABCMeta,
                                             core.WrapperNodeImpl)):

    """
    base node class for providing the standard interface for updating
    """

    children_container = core.DictChildrenContainerSchema(
        cost=core.ChildContainer,
        subtree=core.ChildContainer,
    )
    input_keys = ("subtree_output",)

    def init_state(self, network):
        """
        by default, forward input to cost and subtree, and take output
        from the subtree
        """
        subtree = self._children["subtree"].children
        cost = self._children["cost"].children
        # forward input to both children
        network.forward_input_to(subtree.name)
        network.forward_input_to(cost.name)
        # take output from the subtree
        network.take_output_from(subtree.name, to_key="subtree_output")
        # make it known that the output of this node does NOT depend on the
        # cost node
        network.remove_dependency(cost.name, self.name)

    def new_update_deltas(self, network):
        # compute parameters
        # ---
        if False:
            # only computing for parameters in subtree, not in cost
            subtree = self._children["subtree"].children
            parameters_network = network[subtree.name]
        else:
            # computing for parameters in "subtree" AND "cost"
            # ---
            # example use case: ANRAT - a cost function with parameters
            parameters_network = network
        parameter_vws = parameters_network.find_vws_in_subtree(
            tags=["parameter"])

        # calculate cost
        cost = self._children["cost"].children
        cost_var = network[cost.name].get_variable("default").variable

        # find gradients
        # ---
        # NOTE: gradient computation is factored out to enable future caching
        parameter_variables = [p.variable for p in parameter_vws]
        grads = T.grad(cost_var, parameter_variables)

        # compute update deltas
        return self._new_update_deltas(network, parameter_vws, grads)

    @abc.abstractmethod
    def _new_update_deltas(self, network, parameter_vws, grads):
        pass

# ################################### sgd ###################################


@core.register_node("sgd")
class SGDNode(StandardUpdatesNode):

    """
    node that provides updates via SGD
    """

    hyperparameter_names = ("sgd_learning_rate",
                            "learning_rate")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["sgd_learning_rate",
                                                     "learning_rate"],
                                                    0.1)
        return core.UpdateDeltas({vw.variable: -learning_rate * grad
                                  for vw, grad in zip(parameter_vws,
                                                      grads)})


# ############################ nesterov momentum ############################

@core.register_node("nesterov_momentum")
class NesterovMomentumNode(core.Wrapper1NodeImpl):

    """
    node that transforms all incoming updates to parameters
    in the subtree to use nesterov momentum
    """

    # TODO add way to filter parameters and only apply to a subset
    hyperparameter_names = ("momentum",)

    def mutate_update_deltas(self, network, update_deltas):
        momentum = network.find_hyperparameter(["momentum"], 0.9)
        inits = list(toolz.concat(network.find_hyperparameters(
            ["inits"],
            [])))
        shared_vws = network.find_vws_in_subtree(is_shared=True)
        for vw in shared_vws:
            var = vw.variable
            if var in update_deltas:
                velocity_vw = network.create_vw(
                    "nesterov_momentum_velocity(%s)" % vw.name,
                    shape=vw.shape,
                    is_shared=True,
                    tags={"state"},
                    inits=inits,
                )
                velocity = velocity_vw.variable
                delta = update_deltas[var]
                new_velocity = momentum * velocity + delta
                update_deltas[velocity] = new_velocity - velocity
                update_deltas[var] = delta + momentum * new_velocity


# TODO make a real node
def NAGNode(name, children, learning_rate=None, momentum=None):
    """
    Node for Nesterov's Accelerated Gradient Descent
    """
    subtree = children["subtree"]
    cost = children["cost"]
    if learning_rate is None:
        sgd_kwargs = {}
    else:
        sgd_kwargs = {"learning_rate": learning_rate}
    if momentum is None:
        momentum_kwargs = {}
    else:
        momentum_kwargs = {"momentum": momentum}
    momentum_node = NesterovMomentumNode(name + "_momentum",
                                         subtree,
                                         **momentum_kwargs)
    new_children = {"subtree": momentum_node,
                    "cost": cost}
    return SGDNode(name, new_children, **sgd_kwargs)

# ############################### weight decay ###############################


@core.register_node("weight_decay")
class WeightDecayNode(core.Wrapper1NodeImpl):

    """
    equivalent to L2 loss on the weights
    """

    hyperparameter_names = ("l2_decay",
                            "weight_decay")

    def new_update_deltas(self, network):
        decay = network.find_hyperparameter(["l2_decay",
                                             "weight_decay"],
                                            0)
        if decay == 0:
            # don't add updates if decay is 0
            return super(WeightDecayNode, self).new_update_deltas(network)

        weight_vws = network.find_vws_in_subtree(tags=["weight"])
        weights = [vw.variable for vw in weight_vws]
        return core.UpdateDeltas({w: -decay * w for w in weights})

# ################################### adam ###################################


@core.register_node("adam")
class AdamNode(StandardUpdatesNode):

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
        inits = list(toolz.concat(network.find_hyperparameters(
            ["inits"],
            [])))

        update_deltas = core.UpdateDeltas()

        # keep count state only once
        t_vw = network.create_vw(
            "adam_count",
            shape=(),
            is_shared=True,
            tags={"state"},
            inits=inits,
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
                inits=inits,
            )
            # 2nd moment
            # moving average of squared gradient
            v_vw = network.create_vw(
                "adam_v(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                inits=inits,
            )

            m = m_vw.variable
            v = v_vw.variable

            # new value for 1st moment estimate
            new_m = beta1 * m + (1 - beta1) * grad
            # new value for 2nd moment estimate
            new_v = beta2 * v + (1 - beta2) * T.sqr(grad)

            parameter_delta = - alpha_t * new_m / (T.sqrt(new_v) + epsilon_hat)

            update_deltas[m] = new_m - m
            update_deltas[v] = new_v - v
            update_deltas[parameter_vw.variable] = parameter_delta

        return update_deltas


@core.register_node("adamax")
class AdaMaxNode(StandardUpdatesNode):

    """
    node that provides updates via AdaMax update rule
    based on "Adam: A Method for Stochastic Optimization" v8
    (http://arxiv.org/abs/1412.6980)
    """

    hyperparameter_names = ("adamax_learning_rate",
                            "adamax_alpha",
                            "learning_rate",
                            "adamax_beta1",
                            "beta1",
                            "adamax_beta2",
                            "beta2",
                            "adamax_epsilon",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        # alpha / stepsize / learning rate are all the same thing
        # using alpha because that is what is used in the paper
        alpha = network.find_hyperparameter(["adamax_learning_rate",
                                             "adamax_alpha",
                                             "learning_rate"],
                                            0.002)
        beta1 = network.find_hyperparameter(["adamax_beta1",
                                             "beta1"],
                                            0.9)
        beta2 = network.find_hyperparameter(["adamax_beta2",
                                             "beta2"],
                                            0.999)
        epsilon = network.find_hyperparameter(["adamax_epsilon",
                                               "epsilon"],
                                              1e-8)
        inits = list(toolz.concat(network.find_hyperparameters(
            ["inits"],
            [])))

        update_deltas = core.UpdateDeltas()

        # keep count state only once
        t_vw = network.create_vw(
            "adamax_count",
            shape=(),
            is_shared=True,
            tags={"state"},
            inits=inits,
        )
        t = t_vw.variable
        new_t = t + 1
        update_deltas[t] = new_t - t

        # compute some values only once
        # unbias terms to take into account initializing with 0
        m_unbias_term = 1 - beta1 ** new_t
        alpha_t = alpha / m_unbias_term

        for parameter_vw, grad in zip(parameter_vws, grads):
            # biased 1st moment estimate
            # moving average of gradient
            m_vw = network.create_vw(
                "adamax_m(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                inits=inits,
            )
            # exponentially weighted infinity norm
            u_vw = network.create_vw(
                "adamax_u(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                inits=inits,
            )

            m = m_vw.variable
            u = u_vw.variable

            # new value for 1st moment estimate
            new_m = beta1 * m + (1 - beta1) * grad
            # new value for 2nd moment estimate
            new_u = T.maximum(beta2 * u, abs(grad))

            # NOTE: AdaMax doesn't have the epsilon term in the denominator,
            # but not having it seems to lead to numerical instability
            # (ie. dividing by 0)
            parameter_delta = - alpha_t * new_m / (new_u + epsilon)

            update_deltas[m] = new_m - m
            update_deltas[u] = new_u - u
            update_deltas[parameter_vw.variable] = parameter_delta

        return update_deltas
