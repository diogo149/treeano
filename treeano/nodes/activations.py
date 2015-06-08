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


class ReLUNode(BaseActivationNode):

    def activation(self, network, in_var):
        return relu(in_var.variable)


class SoftmaxNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.nnet.softmax(in_var.variable)


class TanhNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.tanh(in_var.variable)


class SigmoidNode(BaseActivationNode):

    def activation(self, network, in_var):
        return T.nnet.sigmoid(in_var.variable)
