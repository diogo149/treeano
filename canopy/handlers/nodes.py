import toolz
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
        def update_fn(override_hyperparameters):
            return toolz.merge(override_hyperparameters,
                               self.hyperparameters)

        kwargs = toolz.update_in(transforms.fns.network_to_kwargs(network),
                                 ["override_hyperparameters"],
                                 update_fn)
        return treeano.Network(**kwargs)

override_hyperparameters = OverrideHyperparameters
