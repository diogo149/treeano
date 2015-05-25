import abc
import six

from .. import core


class StatelessActivationNode(six.with_metaclass(abc.ABCMeta, core.NodeImpl)):

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


class ReLUNode(StatelessActivationNode):

    def activation(self, network, in_var):
        return relu(in_var.variable)
