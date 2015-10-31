import toolz
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

from . import base
from .. import transforms

fX = theano.config.floatX


class RemoveNodesWithClass(base.NetworkHandlerImpl):

    """
    handler that adds hyperparameters to the network
    """

    def __init__(self, cls):
        self.cls = cls

    def transform_network(self, network):
        return transforms.remove_nodes_with_class(network, self.cls)


remove_nodes_with_class = RemoveNodesWithClass


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

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters

    def transform_network(self, network):
        # FIXME make this a transform

        def update_fn(override_hyperparameters):
            return toolz.merge(override_hyperparameters,
                               self.hyperparameters)

        kwargs = toolz.update_in(transforms.fns.network_to_kwargs(network),
                                 ["override_hyperparameters"],
                                 update_fn)
        return treeano.Network(**kwargs)

override_hyperparameters = OverrideHyperparameters


class UpdateHyperparameters(base.NetworkHandlerImpl):

    """
    handler that replace hyperparameters of a node
    """

    def __init__(self, node_name, **hyperparameters):
        self.node_name = node_name
        self.hyperparameters = hyperparameters

    def transform_network(self, network):
        return transforms.update_hyperparameters(network,
                                                 self.node_name,
                                                 self.hyperparameters)

update_hyperparameters = UpdateHyperparameters


class ScheduleHyperparameter(base.NetworkHandlerImpl):

    def __init__(self,
                 schedule,
                 hyperparameter=None,
                 node_name=None,
                 target_node_name=None,
                 input_key=None,
                 shape=(),
                 dtype=fX):
        """
        WARNING: saves a copy of the previous output

        schedule:
        a function that takes in the current input dictionary and the previous
        output dictionary (or None for the initial value) and returns a new
        value for the hyperparameter

        hyperparameter:
        name of the hyperparameter to provide

        node_name:
        name of the SharedHyperparameterNode to create
        (default: generates one)

        target_node_name:
        name of the node to provide the hyperparameter to
        (default: root node)

        input_key:
        what key to put in the input dict
        (default: generates one)
        """
        self.hyperparameter = hyperparameter
        self.schedule = schedule
        if node_name is None:
            assert hyperparameter is not None
            node_name = "scheduled:%s" % hyperparameter
        self.node_name = node_name
        self.target_node_name = target_node_name
        if input_key is None:
            input_key = node_name
        self.input_key = input_key
        self.shape = shape
        self.dtype = dtype
        self.previous_result_ = None

    def transform_network(self, network):
        # don't add node if it already exists
        if self.node_name in network:
            return network

        assert self.hyperparameter is not None

        if self.target_node_name is None:
            target_node_name = network.root_node.name
        else:
            target_node_name = self.target_node_name

        return transforms.add_parent(
            network=network,
            name=target_node_name,
            parent_constructor=tn.SharedHyperparameterNode,
            parent_name=self.node_name,
            parent_kwargs=dict(
                hyperparameter=self.hyperparameter,
                dtype=self.dtype,
                shape=self.shape,
            ),
        )

    def transform_compile_function_kwargs(self, state, **kwargs):
        assert self.input_key not in kwargs["inputs"]
        kwargs["inputs"][self.input_key] = (self.node_name, "hyperparameter")
        return kwargs

    def call(self, fn, in_dict, *args, **kwargs):
        assert self.input_key not in in_dict
        hyperparameter_value = self.schedule(in_dict, self.previous_result_)
        in_dict[self.input_key] = np.array(hyperparameter_value,
                                           dtype=self.dtype)
        res = fn(in_dict, *args, **kwargs)
        self.previous_result_ = res
        return res

schedule_hyperparameter = ScheduleHyperparameter


class UseScheduledHyperparameter(base.NetworkHandlerImpl):

    """
    allows a network to use the scheduled hyperparameter of a different
    network

    use case:
    - having a validation network use the same parameter as a training
      network
    """

    def __init__(self, schedule_hyperparameter_handler):
        assert isinstance(schedule_hyperparameter_handler,
                          ScheduleHyperparameter)
        self.shh = schedule_hyperparameter_handler

    def transform_network(self, network):
        return self.shh.transform_network(network)

    def transform_compile_function_kwargs(self, state, **kwargs):
        return self.shh.transform_compile_function_kwargs(state, **kwargs)

use_scheduled_hyperparameter = UseScheduledHyperparameter
