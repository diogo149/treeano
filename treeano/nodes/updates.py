"""
nodes that provide updates for shared variables
"""
import abc

import six
import numpy as np
import theano
import theano.tensor as T

from .. import core
from .. import inits


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
        children = self.raw_children()
        subtree = children["subtree"]
        cost = children["cost"]
        # forward input to both children
        network.forward_input_to(subtree.name)
        network.forward_input_to(cost.name)
        # take output from the subtree
        network.take_output_from(subtree.name, to_key="subtree_output")
        # make it known that the output of this node does NOT depend on the
        # cost node
        network.remove_dependency(cost.name, self.name)

    def new_update_deltas(self, network):
        children = self.raw_children()
        # compute parameters
        # ---
        if False:
            # only computing for parameters in subtree, not in cost
            subtree = children["subtree"]
            parameters_network = network[subtree.name]
        else:
            # computing for parameters in "subtree" AND "cost"
            # ---
            # example use case: ANRAT - a cost function with parameters
            parameters_network = network
        parameter_vws = parameters_network.find_vws_in_subtree(
            tags=["parameter"])

        # calculate cost
        cost = children["cost"]
        cost_var = network[cost.name].get_vw("default").variable

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


# ############################ momentum ############################

@core.register_node("momentum")
class MomentumNode(core.Wrapper1NodeImpl):

    """
    node that transforms all incoming updates to parameters
    in the subtree to use momentum
    """

    # TODO add way to filter parameters and only apply to a subset
    hyperparameter_names = ("momentum",)

    def mutate_update_deltas(self, network, update_deltas):
        momentum = network.find_hyperparameter(["momentum"], 0.9)
        shared_vws = network.find_vws_in_subtree(is_shared=True)
        for vw in shared_vws:
            var = vw.variable
            if var in update_deltas:
                velocity = network.create_vw(
                    "momentum_velocity(%s)" % vw.name,
                    shape=vw.shape,
                    is_shared=True,
                    tags={"state"},
                    default_inits=[],
                ).variable
                delta = update_deltas[var]
                new_velocity = momentum * velocity + delta
                update_deltas[velocity] = new_velocity - velocity
                update_deltas[var] = new_velocity


@core.register_node("momentum_sgd")
class MomentumSGDNode(core.WrapperNodeImpl):

    """
    node that provides updates via SGD + momentum
    """

    children_container = core.DictChildrenContainerSchema(
        cost=core.ChildContainer,
        subtree=core.ChildContainer,
    )

    hyperparameter_names = (SGDNode.hyperparameter_names
                            + MomentumNode.hyperparameter_names)

    def architecture_children(self):
        children = self.raw_children()
        momentum_node = MomentumNode(self.name + "_momentum",
                                     children["subtree"])
        new_children = {"subtree": momentum_node,
                        "cost": children["cost"]}
        return [SGDNode(self.name + "_sgd", new_children)]


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
        shared_vws = network.find_vws_in_subtree(is_shared=True)
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
                update_deltas[var] = delta + momentum * new_velocity


@core.register_node("nesterovs_accelerated_gradient")
class NesterovsAcceleratedGradientNode(core.WrapperNodeImpl):

    """
    node that provides updates via SGD Nesterov's Accelerated Gradient Descent
    """

    children_container = core.DictChildrenContainerSchema(
        cost=core.ChildContainer,
        subtree=core.ChildContainer,
    )

    hyperparameter_names = (SGDNode.hyperparameter_names
                            + NesterovMomentumNode.hyperparameter_names)

    def architecture_children(self):
        children = self.raw_children()
        momentum_node = NesterovMomentumNode(self.name + "_momentum",
                                             children["subtree"])
        new_children = {"subtree": momentum_node,
                        "cost": children["cost"]}
        return [SGDNode(self.name + "_sgd", new_children)]

# alias
NAGNode = NesterovsAcceleratedGradientNode


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

        update_deltas = core.UpdateDeltas()

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

        update_deltas = core.UpdateDeltas()

        # keep count state only once
        t_vw = network.create_vw(
            "adamax_count",
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
                default_inits=[],
            )
            # exponentially weighted infinity norm
            u_vw = network.create_vw(
                "adamax_u(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
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


# ################################# ADADELTA #################################


@core.register_node("adadelta")
class ADADELTANode(StandardUpdatesNode):

    """
    from "ADADELTA: An Adaptive Learning Rate Method"
    (http://arxiv.org/abs/1212.5701)
    """

    hyperparameter_names = ("rho",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        rho = network.find_hyperparameter(["rho"], 0.95)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-6)

        update_deltas = core.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # exponential moving average of deltas squared
            d2_avg = network.create_vw(
                "adadelta_deltas_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable
            # exponential moving average of gradients squared
            g2_avg = network.create_vw(
                "adadelta_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated gradients squared
            new_g2_avg = rho * g2_avg + (1 - rho) * grad ** 2

            # calculate update
            deltas = -(grad * T.sqrt(d2_avg + epsilon)
                       / T.sqrt(new_g2_avg + epsilon))

            # updated deltas squared
            new_d2_avg = rho * d2_avg + (1 - rho) * deltas ** 2

            update_deltas[g2_avg] = new_g2_avg - g2_avg
            update_deltas[d2_avg] = new_d2_avg - d2_avg
            update_deltas[parameter_vw.variable] = deltas

        return update_deltas

# ################################# ADAGRAD #################################


@core.register_node("adagrad")
class ADAGRADNode(StandardUpdatesNode):

    """
    from "Adaptive subgradient methods for online learning and stochastic
    optimization"
    """

    hyperparameter_names = ("learning_rate",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 1e-3)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-6)

        update_deltas = core.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # sum of gradients squared
            g2_sum = network.create_vw(
                "adagrad_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated gradients squared
            new_g2_sum = g2_sum + grad ** 2

            # calculate update
            deltas = -learning_rate * grad / T.sqrt(new_g2_sum + epsilon)

            update_deltas[g2_sum] = new_g2_sum - g2_sum
            update_deltas[parameter_vw.variable] = deltas

        return update_deltas

# ################################# RMSprop #################################


@core.register_node("rmsprop")
class RMSPropNode(StandardUpdatesNode):

    """
    from Neural Networks for Machine Learning Coursera class (lecture 6.5)
    """

    hyperparameter_names = ("learning_rate",
                            "rho",
                            "epsilon")

    def _new_update_deltas(self, network, parameter_vws, grads):
        learning_rate = network.find_hyperparameter(["learning_rate"], 1e-2)
        rho = network.find_hyperparameter(["rho"], 0.9)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)

        update_deltas = core.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):
            # exponential moving average of gradients squared
            g2_avg = network.create_vw(
                "rmsprop_gradients_squared(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            # updated gradients squared
            new_g2_avg = rho * g2_avg + (1 - rho) * T.sqr(grad)

            # calculate update
            deltas = -learning_rate * grad / T.sqrt(new_g2_avg + epsilon)

            update_deltas[g2_avg] = new_g2_avg - g2_avg
            update_deltas[parameter_vw.variable] = deltas

        return update_deltas


# ################################## Rprop ##################################


@core.register_node("rprop")
class RpropNode(StandardUpdatesNode):

    """
    from "A Direct Adaptive Method for Faster Backpropagation Learning:
    The RPROP Algorithm"

    supported rprop_type's:
    "Rprop" or "Rprop+": Rprop w/ weight-backtracking (ie. original Rprop)
    "iRprop-": improved Rprop w/o weight-backtracking
    """

    hyperparameter_names = ("initial_step",
                            "learning_rate",
                            "eta_plus",
                            "eta_minus",
                            "step_min",
                            "step_max",
                            "rprop_type")

    def _new_update_deltas(self, network, parameter_vws, grads):
        initial_step = network.find_hyperparameter(["initial_step",
                                                    "learning_rate"], 0.0125)
        eta_plus = network.find_hyperparameter(["eta_plus"], 1.2)
        eta_minus = network.find_hyperparameter(["eta_minus"], 0.5)
        step_min = network.find_hyperparameter(["step_min"], 0)
        step_max = network.find_hyperparameter(["step_max"], 50)
        rprop_type = network.find_hyperparameter(["rprop_type"],
                                                 "iRprop-")

        compute_prev_delta = rprop_type in {"Rprop", "Rprop+"}
        plus_delta = eta_plus - 1
        minus_delta = eta_minus - 1

        update_deltas = core.UpdateDeltas()
        for parameter_vw, grad in zip(parameter_vws, grads):

            step = network.create_vw(
                "rprop_step(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[inits.ConstantInit(initial_step)],
            ).variable

            prev_grad = network.create_vw(
                "rprop_prev_grad(%s)" % parameter_vw.name,
                shape=parameter_vw.shape,
                is_shared=True,
                tags={"state"},
                default_inits=[],
            ).variable

            if compute_prev_delta:
                prev_delta = network.create_vw(
                    "rprop_prev_delta(%s)" % parameter_vw.name,
                    shape=parameter_vw.shape,
                    is_shared=True,
                    tags={"state"},
                    default_inits=[],
                ).variable

            grad_product = grad * prev_grad
            pos_mask = grad_product > 0
            neg_mask = grad_product < 0
            non_neg_mask = 1 - neg_mask

            # change step size based on sign change of gradient
            step_delta = (plus_delta * pos_mask
                          + minus_delta * neg_mask) * step
            new_step = T.clip(step + step_delta, step_min, step_max)

            # update in direction of negative gradient
            parameter_delta = -T.sgn(grad) * new_step

            if rprop_type in {"Rprop", "Rprop+"}:
                # if grad product is negative:
                # - revert previous update
                # - set gradient to 0
                parameter_delta = (non_neg_mask * parameter_delta
                                   - neg_mask * prev_delta)
                grad = non_neg_mask * grad
            elif rprop_type == "Rprop-":
                # do nothing
                pass
            elif rprop_type == "iRprop+":
                raise NotImplementedError()
            elif rprop_type == "iRprop-":
                # if grad product is negative:
                # - set gradient to 0
                grad = non_neg_mask * grad
            else:
                raise ValueError("Incorrect rprop_type: %s" % rprop_type)

            update_deltas[step] = new_step - step
            update_deltas[prev_grad] = grad - prev_grad
            update_deltas[parameter_vw.variable] = parameter_delta
            if compute_prev_delta:
                update_deltas[prev_delta] = parameter_delta - prev_delta

        return update_deltas
