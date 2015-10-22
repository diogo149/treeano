import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils


def _is_pos(var):
    return (var > 0).astype(var.dtype)


def _is_neg(var):
    return (var < 0).astype(var.dtype)


def _or(var1, var2):
    return treeano.utils.maximum(var1, var2)


def _and(var1, var2):
    return var1 * var2


class _LinearGrad(treeano.sandbox.utils.OverwriteGrad):

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        return (grd,)

linear_grad_relu = _LinearGrad(treeano.utils.rectify)


@treeano.register_node("linear_grad_relu")
class LinearGradReLUNode(tn.BaseActivationNode):

    def activation(self, network, in_vw):
        return linear_grad_relu(in_vw.variable)


class _LeakyGrad(treeano.sandbox.utils.OverwriteGrad):

    CACHED = {}

    def __init__(self, alpha):
        self.alpha_ = alpha
        super(_LeakyGrad, self).__init__(fn=treeano.utils.rectify)

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        return (grd * T.clip(_is_pos(inp), self.alpha_, 1),)

    @staticmethod
    def get_cached(alpha):
        assert treeano.utils.is_number(alpha)
        cached = _ConservedLeakyPositivePushing.CACHED
        if alpha not in cached:
            cached[alpha] = _ConservedLeakyPositivePushing(alpha)
        return cached[alpha]


@treeano.register_node("leaky_grad_relu")
class LeakyGradReLUNode(tn.BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"])
        return _LeakyGrad.get_cached(alpha)(in_vw.variable)


class _PositivePushing(treeano.sandbox.utils.OverwriteGrad):

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        return (grd * _or(_is_pos(inp), _is_pos(grd)),)

positive_pushing_relu = _PositivePushing(treeano.utils.rectify)


@treeano.register_node("positive_pushing_relu")
class PositivePushingReLUNode(tn.BaseActivationNode):

    def activation(self, network, in_vw):
        return positive_pushing_relu(in_vw.variable)


class _LeakyPositivePushing(treeano.sandbox.utils.OverwriteGrad):

    CACHED = {}

    def __init__(self, alpha):
        self.alpha_ = alpha
        super(_LeakyPositivePushing, self).__init__(fn=treeano.utils.rectify)

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        return (grd * _or(_is_pos(inp), self.alpha_ * _is_pos(grd)),)

    @staticmethod
    def get_cached(alpha):
        assert treeano.utils.is_number(alpha)
        cached = _LeakyPositivePushing.CACHED
        if alpha not in cached:
            cached[alpha] = _LeakyPositivePushing(alpha)
        return cached[alpha]


@treeano.register_node("leaky_positive_pushing_relu")
class LeakyPositivePushingReLUNode(tn.BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"])
        return _LeakyPositivePushing.get_cached(alpha)(in_vw.variable)


class _IndependentLeakyPositivePushing(treeano.sandbox.utils.OverwriteGrad):

    CACHED = {}

    def __init__(self, alpha):
        self.alpha_ = alpha
        super(_IndependentLeakyPositivePushing, self).__init__(
            fn=treeano.utils.rectify)

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        return (grd * ((1 - self.alpha_) * _is_pos(inp)
                       + self.alpha_ * _is_pos(grd)),)

    @staticmethod
    def get_cached(alpha):
        assert treeano.utils.is_number(alpha)
        cached = _IndependentLeakyPositivePushing.CACHED
        if alpha not in cached:
            cached[alpha] = _IndependentLeakyPositivePushing(alpha)
        return cached[alpha]


@treeano.register_node("independent_leaky_positive_pushing_relu")
class IndependentLeakyPositivePushingReLUNode(tn.BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"])
        return _IndependentLeakyPositivePushing.get_cached(alpha)(
            in_vw.variable)


class _ConservedLeakyPositivePushing(treeano.sandbox.utils.OverwriteGrad):

    CACHED = {}

    def __init__(self, alpha):
        self.alpha_ = alpha
        super(_ConservedLeakyPositivePushing, self).__init__(
            fn=treeano.utils.rectify)

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        p_inp = _is_pos(inp)
        p_grd = _is_pos(grd)
        return (grd * (p_inp
                       + self.alpha_ * p_grd
                       - 2 * self.alpha_ * _or(p_inp, p_grd)),)

    @staticmethod
    def get_cached(alpha):
        assert treeano.utils.is_number(alpha)
        cached = _ConservedLeakyPositivePushing.CACHED
        if alpha not in cached:
            cached[alpha] = _ConservedLeakyPositivePushing(alpha)
        return cached[alpha]


@treeano.register_node("conserved_leaky_positive_pushing_relu")
class ConservedLeakyPositivePushingReLUNode(tn.BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"])
        return _ConservedLeakyPositivePushing.get_cached(alpha)(
            in_vw.variable)
