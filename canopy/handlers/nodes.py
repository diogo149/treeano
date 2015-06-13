import treeano

from . import base
from .. import transforms


class WithHyperparameters(base.NetworkHandlerImpl):

    """
    handler that adds hyperparameters to the network
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.hyperparameters = kwargs

    def transform_network(self, network):
        return transforms.add_hyperparameters(network,
                                              self.name,
                                              self.hyperparameters)


with_hyperparameters = WithHyperparameters


class OverrideHyperparameters(base.NetworkHandlerImpl):

    """
    handler that adds override hyperparameters to the network
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs

    def transform_network(self, network):
        new_override_hyperparameters = dict(network.override_hyperparameters)
        new_override_hyperparameters.update(self.hyperparameters)
        return treeano.Network(
            network.root_node,
            override_hyperparameters=new_override_hyperparameters,
            default_hyperparameters=network.default_hyperparameters,
        )

override_hyperparameters = OverrideHyperparameters
