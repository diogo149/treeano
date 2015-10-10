import abc
import six

import theano.tensor as T

from .. import core
from .. import utils


class BaseActivationNode(six.with_metaclass(abc.ABCMeta, core.NodeImpl)):

    """
    base node class for activation functions
    """

    def compute_output(self, network, in_vw):
        new_var = self.activation(network, in_vw)
        network.create_vw(
            "default",
            variable=new_var,
            shape=in_vw.shape,
            tags={"output"},
        )

    @abc.abstractmethod
    def activation(self, network, in_vw):
        pass


@core.register_node("relu")
class ReLUNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return utils.rectify(in_vw.variable)


@core.register_node("softmax")
class SoftmaxNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.nnet.softmax(in_vw.variable)


@core.register_node("stable_softmax")
class StableSoftmaxNode(BaseActivationNode):

    hyperparameter_names = ("axis",)

    def activation(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"], 1)
        return utils.stable_softmax(in_vw.variable, axis=axis)


@core.register_node("tanh")
class TanhNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.tanh(in_vw.variable)


@core.register_node("scaled_tanh")
class ScaledTanhNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return 1.7159 * T.tanh(in_vw.variable * (2.0 / 3.0))


@core.register_node("sigmoid")
class SigmoidNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.nnet.sigmoid(in_vw.variable)


@core.register_node("resqrt")
class ReSQRTNode(BaseActivationNode):

    """
    rectified shifted square root
    from "Author Identification using Multi-headed Recurrent Neural Networks"
    http://arxiv.org/abs/1506.04891
    """

    def activation(self, network, in_vw):
        r = utils.rectify(in_vw.variable)
        return T.sqrt(r + 1) - 1


@core.register_node("abs")
class AbsNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return abs(in_vw.variable)


@core.register_node("leaky_relu")
class LeakyReLUNode(BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"],
                                            0.01)
        return utils.rectify(in_vw.variable, negative_coefficient=alpha)


@core.register_node("very_leaky_relu")
class VeryLeakyReLUNode(BaseActivationNode):

    hyperparameter_names = ("leak_alpha",
                            "alpha")

    def activation(self, network, in_vw):
        alpha = network.find_hyperparameter(["leak_alpha",
                                             "alpha"],
                                            1. / 3)
        return utils.rectify(in_vw.variable, negative_coefficient=alpha)
