"""
nodes specific to dealing with hyperparameters
"""

# FIXME add HyperparameterNode to this file

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T

from .. import utils
from .. import core

fX = theano.config.floatX


@core.register_node("variable_hyperparameter")
class VariableHyperparameterNode(core.Wrapper1NodeImpl):

    """
    provides a theano variable for the given hyperparameter to its children
    """

    hyperparameter_names = ("hyperparameter", "dtype", "shape")

    def init_state(self, network):
        # perform default
        super(VariableHyperparameterNode, self).init_state(network)
        # also create variable
        dtype = network.find_hyperparameter(["dtype"], fX)
        shape = network.find_hyperparameter(["shape"], ())
        network.create_vw(
            "hyperparameter",
            shape=shape,
            dtype=dtype,
            is_shared=False,
            tags={"hyperparameter", "monitor"},
        )

    def get_hyperparameter(self, network, name):
        if name == "hyperparameter":
            # prevent infinite loop
            return super(VariableHyperparameterNode, self).get_hyperparameter(
                network, name)

        rel_network = network[self.name]
        hyperparameter = rel_network.find_hyperparameter(["hyperparameter"])
        if name == hyperparameter:
            return rel_network.get_vw("hyperparameter").variable
        else:
            return super(VariableHyperparameterNode, self).get_hyperparameter(
                network, name)


@core.register_node("shared_hyperparameter")
class SharedHyperparameterNode(core.Wrapper1NodeImpl):

    """
    provides a theano variable for the given hyperparameter to its children,
    but also keeps the last value as a shared variable

    use case:
    - loading a network that was saved while the hyperparameter was changing
    """
    # FIXME a lot copy pasted from VariableHyperparameterNode

    hyperparameter_names = ("hyperparameter", "dtype", "shape", "inits")

    def init_state(self, network):
        # perform default
        super(SharedHyperparameterNode, self).init_state(network)
        # also create variable
        dtype = network.find_hyperparameter(["dtype"], fX)
        shape = network.find_hyperparameter(["shape"], ())
        # TODO take in optional initial value instead of dtype/shape
        raw_vw = network.create_vw(
            "raw_hyperparameter",
            shape=shape,
            dtype=dtype,
            is_shared=True,
            default_inits=[],
            tags={"state"},
        )
        # create a copy of the shared variable, so we can use this to
        # update the variable
        network.copy_vw(
            name="hyperparameter",
            previous_vw=raw_vw,
            tags={"hyperparameter", "monitor"},
        )

    def new_update_deltas(self, network):
        raw = network.get_vw("raw_hyperparameter").variable
        actual = network.get_vw("hyperparameter").variable
        return core.UpdateDeltas.from_updates({raw: actual})

    def get_hyperparameter(self, network, name):
        if name == "hyperparameter":
            # prevent infinite loop
            return super(SharedHyperparameterNode, self).get_hyperparameter(
                network, name)

        rel_network = network[self.name]
        hyperparameter = rel_network.find_hyperparameter(["hyperparameter"])
        if name == hyperparameter:
            return rel_network.get_vw("hyperparameter").variable
        else:
            return super(SharedHyperparameterNode, self).get_hyperparameter(
                network, name)


@core.register_node("output_hyperparameter")
class OutputHyperparameterNode(core.NodeImpl):

    """
    outputs a hyperparameter of the node
    """

    hyperparameter_names = ("hyperparameter", "shape")
    input_keys = ()

    def compute_output(self, network):
        hyperparameter_name = network.find_hyperparameter(["hyperparameter"])
        # TODO add default hyperparameter
        res = network.find_hyperparameter([hyperparameter_name])
        if utils.is_number(res):
            var = T.constant(res)
            shape = ()
        elif utils.is_ndarray(res):
            var = T.constant(res)
            shape = res.shape
        elif utils.is_shared_variable(res):
            var = res
            shape = res.get_value().shape
        elif utils.is_nonshared_variable(res):
            var = res
            if res.ndim == 0:
                shape = ()
            else:
                shape = network.find_hyperparameter(["shape"])
        else:
            raise ValueError("Unknown hyperparameter type of %s" % res)

        network.create_vw(
            "default",
            variable=var,
            shape=shape,
            tags={"output"},
        )
