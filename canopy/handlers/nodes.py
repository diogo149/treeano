import toolz
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

from . import base
from .. import transforms

fX = theano.config.floatX


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
        # FIXME make this a transform

        def update_fn(override_hyperparameters):
            return toolz.merge(override_hyperparameters,
                               self.hyperparameters)

        kwargs = toolz.update_in(transforms.fns.network_to_kwargs(network),
                                 ["override_hyperparameters"],
                                 update_fn)
        return treeano.Network(**kwargs)

override_hyperparameters = OverrideHyperparameters


class ScheduledHyperparameter(base.NetworkHandlerImpl):

    def __init__(self,
                 hyperparameter,
                 schedule,
                 node_name=None,
                 target_node_name=None,
                 input_key=None,
                 shape=(),
                 dtype=fX):
        """
        WARNING: saves a copy of the previous output

        hyperparameter:
        name of the hyperparameter to provide

        schedule:
        a function that takes in the current input dictionary and the previous
        output dictionary (or None for the initial value) and returns a new
        value for the hyperparameter

        node_name:
        name of the VariableHyperparameterNode to create
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
        self.node_name = node_name
        self.target_node_name = target_node_name
        self.input_key = input_key
        self.shape = shape
        self.dtype = dtype
        self.previous_result_ = None

    def transform_network(self, network):
        if self.target_node_name is None:
            self.target_node_name_ = network.root_node.name
        else:
            self.target_node_name_ = self.target_node_name

        if self.node_name is None:
            self.node_name_ = "%s_%s" % (self.target_node_name_,
                                         self.hyperparameter)
        else:
            self.node_name_ = self.node_name

        return transforms.add_parent(
            network=network,
            name=self.target_node_name_,
            parent_constructor=tn.VariableHyperparameterNode,
            parent_name=self.node_name_,
            parent_kwargs=dict(
                hyperparameter=self.hyperparameter,
                dtype=self.dtype,
                shape=self.shape,
            ),
        )

    def transform_compile_function_kwargs(self, state, **kwargs):
        if self.input_key is None:
            self.input_key_ = self.node_name_
        else:
            self.input_key_ = self.input_key

        assert self.input_key_ not in kwargs["inputs"]
        kwargs["inputs"][self.input_key_] = (self.node_name_, "hyperparameter")
        return kwargs

    def call(self, fn, in_dict, *args, **kwargs):
        hyperparameter_value = self.schedule(in_dict, self.previous_result_)
        in_dict[self.input_key_] = np.array(hyperparameter_value,
                                            dtype=self.dtype)
        res = fn(in_dict, *args, **kwargs)
        self.previous_result_ = res
        return res

schedule_hyperparameter = ScheduledHyperparameter
