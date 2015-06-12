from . import base


class WithHyperparameters(base.NetworkHandlerImpl):

    """
    handler that adds hyperparameters to the network
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs

    def transform_network(self, network):
        # FIXME
        pass


with_hyperparameters = WithHyperparameters


class OverrideHyperparameters(base.NetworkHandlerImpl):

    """
    handler that adds override hyperparameters to the network
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs

    def transform_network(self, network):
        # FIXME
        pass

override_hyperparameters = OverrideHyperparameters
