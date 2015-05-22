"""
some not-actually-useful nodes (if they are, move them elsewhere) mostly
for tests and code samples
"""

from .. import core


@core.register_node("add_constant")
class AddConstantNode(core.NodeImpl):

    """
    adds a constant value to its input
    """

    hyperparameter_names = ("value", )

    def compute_output(self, network, in_var):
        value = network.find_hyperparameter(["value"])
        network.create_variable(
            name="default",
            variable=in_var.variable + value,
            shape=in_var.shape,
            tags={"output"},
        )


@core.register_node("multiply_constant")
class MultiplyConstantNode(core.NodeImpl):

    """
    adds a constant value to its input
    """

    hyperparameter_names = ("value", )

    def compute_output(self, network, in_var):
        value = network.find_hyperparameter(["value"])
        network.create_variable(
            name="default",
            variable=in_var.variable * value,
            shape=in_var.shape,
            tags={"output"},
        )
