import abc
import six

import theano.tensor as T

from .. import core


class BaseActivationNode(six.with_metaclass(abc.ABCMeta, core.NodeImpl)):

    """
    base node class for activation functions
    """

    def compute_output(self, network, in_var):
        new_var = self.activation(network, in_var)
        network.create_variable(
            "default",
            variable=new_var,
            shape=in_var.shape,
            tags={"output"},
        )

    @abc.abstractmethod
    def activation(self, network, in_var):
        pass


def relu(x):
    return 0.5 * (x + abs(x))


@core.register_node("relu")
class ReLUNode(BaseActivationNode):

    def activation(self, network, in_var):
        return relu(in_var.variable)


@core.register_node("softmax")
class SoftmaxNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.nnet.softmax(in_var.variable)


@core.register_node("tanh")
class TanhNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.tanh(in_var.variable)


@core.register_node("scaled_tanh")
class ScaledTanhNode(BaseActivationNode):

    def activation(self, network, in_var):
        return 1.7159 * T.tanh(in_var.variable * (2.0 / 3.0))


@core.register_node("sigmoid")
class SigmoidNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.nnet.sigmoid(in_var.variable)


@core.register_node("resqrt")
class ReSQRTNode(BaseActivationNode):

    """
    rectified shifted square root
    from "Author Identification using Multi-headed Recurrent Neural Networks"
    http://arxiv.org/abs/1506.04891
    """

    def activation(self, network, in_var):
        r = relu(in_var.variable)
        return T.sqrt(r + 1) - 1
