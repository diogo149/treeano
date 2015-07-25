"""
nodes that provide updates for shared variables
"""
import abc

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
        parameters = parameters_network.find_vws_in_subtree(tags=["parameter"])

        # calculate cost
        cost = self._children["cost"].children
        cost_var = network[cost.name].get_variable("default").variable

        # find gradients
        # ---
        # NOTE: gradient computation is factored out to enable future caching
        parameter_variables = [p.variable for p in parameters]
        grads = T.grad(cost_var, parameter_variables)

        # compute update deltas
        return self._new_update_deltas(network, parameters, grads)

    @abc.abstractmethod
    def _new_update_deltas(self, network, parameters, grads):
        pass

# ################################### sgd ###################################


@core.register_node("sgd")
class SGDNode(StandardUpdatesNode):

    """
    node that provides updates via SGD
    """

    hyperparameter_names = ("sgd_learning_rate",
                            "learning_rate")

    def _new_update_deltas(self, network, parameters, grads):
        learning_rate = network.find_hyperparameter(["sgd_learning_rate",
                                                     "learning_rate"])
        parameter_variables = [p.variable for p in parameters]
        return core.UpdateDeltas({param: -learning_rate * grad
                                  for param, grad in zip(parameter_variables,
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
        # FIXME get/set inits
        shared_vws = network.find_vws_in_subtree(is_shared=True)
        for vw in shared_vws:
            var = vw.variable
            if var in update_deltas:
                velocity_vw = network.create_variable(
                    "velocity(%s)" % vw.name,
                    shape=vw.shape,
                    is_shared=True,
                    tags={"state"},
                )
                velocity = velocity_vw.variable
                delta = update_deltas[var]
                new_velocity = momentum * velocity + delta
                update_deltas[velocity] = new_velocity - velocity
                update_deltas[var] = delta + momentum * new_velocity


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

# ################################### adam ###################################


def adam_v4(all_grads,
            all_params,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            lambda_=1 - 1e-8):
    """
    based on Adam update rule http://arxiv.org/abs/1412.6980
    (v4 or v5, which is the same as v4)
    """
    updates = []

    # alpha / stepsize / learning rate are all the same thing
    # using alpha because that is what is used in the paper
    alpha = learning_rate

    t = theano.shared(np.array(0., dtype=theano.config.floatX))
    t_next = t + 1
    beta1_t = beta1 * lambda_ ** t

    # compute some values only once
    # unbias terms to take into account initializing with 0
    m_unbias_term = 1 - beta1 ** t_next
    v_unbias_term = T.sqrt(1 - beta2 ** t_next)
    epsilon_hat = epsilon * v_unbias_term
    alpha_t = alpha * v_unbias_term / m_unbias_term

    for param, grad in zip(all_params, all_grads):
        # 1st moment
        mparam = theano.shared(np.zeros(param.get_value().shape,
                                        dtype=theano.config.floatX))
        # 2nd moment
        vparam = theano.shared(np.zeros(param.get_value().shape,
                                        dtype=theano.config.floatX))

        # new value for 1st moment estimate
        m = beta1_t * mparam + (1 - beta1_t) * grad
        # new value for 2nd moment estimate
        v = beta2 * vparam + (1 - beta2) * T.sqr(grad)

        param_next = param - alpha_t * m / (T.sqrt(v) + epsilon_hat)

        updates.append((mparam, m))
        updates.append((vparam, v))
        updates.append((param, param_next))

    updates.append((t, t_next))
    return updates


@core.register_node("adam")
class AdamNode(StandardUpdatesNode):

    """
    node that provides updates via Adam update rule
    """

    hyperparameter_names = ("adam_learning_rate",
                            "adam_alpha",
                            "learning_rate",
                            "adam_beta1",
                            "beta1")

    def _new_update_deltas(self, network, parameters, grads):
        learning_rate = network.find_hyperparameter(["adam_learning_rate",
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
        lambda_ = network.find_hyperparameter(["adam_lambda",
                                               "lambda_",
                                               "lambda"],
                                              1 - 1e-8)
        parameter_variables = [p.variable for p in parameters]
        updates = adam_v4(grads,
                          parameter_variables,
                          learning_rate=learning_rate,
                          beta1=beta1,
                          beta2=beta2,
                          epsilon=epsilon,
                          lambda_=lambda_)
        return core.UpdateDeltas.from_updates(updates)
