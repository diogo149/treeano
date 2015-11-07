"""
a bunch of simple, composable nodes - not necessarily simple in their
implementation, but simple in that they do a single thing
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T

from .. import utils
from .. import core


@core.register_node("reference")
class ReferenceNode(core.NodeImpl):

    """
    provides dependencies into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them

    reference:
    node name to take output from as input to the reference node
    """

    hyperparameter_names = ("reference",)
    input_keys = ("reference_input",)

    def init_long_range_dependencies(self, network):
        network.take_output_from(network.find_hyperparameter(["reference"]),
                                 to_key="reference_input")


@core.register_node("send_to")
class SendToNode(core.NodeImpl):

    """
    sends the input of the node into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them

    reference:
    node name to take output from as input to the reference node

    to_key:
    input key for the reference node of the SendToNode's output
    """

    hyperparameter_names = ("send_to_reference",
                            "reference",
                            "to_key")

    def init_long_range_dependencies(self, network):
        network.forward_output_to(
            network.find_hyperparameter(["send_to_reference",
                                         "reference"]),
            to_key=network.find_hyperparameter(["to_key"],
                                               "default"))


@core.register_node("hyperparameter")
class HyperparameterNode(core.Wrapper1NodeImpl):

    """
    for providing hyperparameters to a subtree
    """

    def __init__(self, name, children, **hyperparameters):
        # set hyperparameter keys to be all passed in keys
        self.hyperparameter_names = tuple(hyperparameters.keys())
        # override init to allow for using keyword arguments
        super(HyperparameterNode, self).__init__(name,
                                                 children,
                                                 **hyperparameters)


@core.register_node("input")
class InputNode(core.NodeImpl):

    """
    an entry point into the network
    """

    hyperparameter_names = ("input_shape",
                            "shape",
                            "input_dtype",
                            "dtype",
                            "input_broadcastable",
                            "broadcastable")
    input_keys = ()

    def compute_output(self, network):
        network.create_vw(
            name="default",
            shape=network.find_hyperparameter(["input_shape",
                                               "shape"]),
            dtype=network.find_hyperparameter(["input_dtype",
                                               "dtype"],
                                              theano.config.floatX),
            broadcastable=network.find_hyperparameter(["input_broadcastable",
                                                       "broadcastable"],
                                                      None),
            is_shared=False,
            tags=["input"],
        )


@core.register_node("identity")
class IdentityNode(core.NodeImpl):

    """
    returns input
    """
    # NOTE: default implementation of NodeImpl is Identity


@core.register_node("constant")
class ConstantNode(core.NodeImpl):

    """
    node that returns a constant value
    """

    hyperparameter_names = ("constant_value", "value")
    input_keys = ()

    def compute_output(self, network):
        value = network.find_hyperparameter(["constant_value", "value"])
        if utils.is_nonshared_variable(value):
            variable = value
            shape = None
        elif utils.is_shared_variable(value):
            variable = value
            shape = value.get_value().shape
        else:
            variable = T.constant(value)
            shape = value.shape if hasattr(value, "shape") else ()
        network.create_vw(
            name="default",
            variable=variable,
            shape=shape,
            tags={"output"},
        )


@core.register_node("add_bias")
class AddBiasNode(core.NodeImpl):

    """
    node that adds a bias parameter to its input

    takes in either a list of axes (as key broadcastable_axes) to broadcast
    over, or as a tuple of booleans
    """

    hyperparameter_names = ("bias_inits",
                            "inits",
                            "broadcastable_axes",
                            "broadcastable",)

    def compute_output(self, network, in_vw):
        # gather hyperparameters
        broadcastable = network.find_hyperparameter(["broadcastable"],
                                                    None)
        broadcastable_axes = network.find_hyperparameter(
            ["broadcastable_axes"],
            None)
        batch_axis = network.find_hyperparameter(["batch_axis"])
        # have broadcastable as a tuple take precedence over broadcastable_axes
        if broadcastable is None:
            if broadcastable_axes is None:
                if batch_axis is None:
                    # no minibatch axis = no default broadcasting
                    broadcastable_axes = []
                elif batch_axis >= in_vw.ndim:
                    # scalar input = no broadcasting
                    broadcastable_axes = []
                else:
                    # by default, broadcast over minibatch axis, if any
                    broadcastable_axes = [batch_axis]
            broadcastable = [False] * in_vw.ndim
            for axis in broadcastable_axes:
                broadcastable[axis] = True

        assert len(broadcastable) == in_vw.ndim
        shape = tuple([1 if is_broadcastable else size
                       for is_broadcastable, size in zip(broadcastable,
                                                         in_vw.shape)])
        b = network.create_vw(
            name="bias",
            is_shared=True,
            shape=shape,
            tags={"parameter", "bias"},
            default_inits=[],
            default_inits_hyperparameters=["bias_inits", "inits"],
        )
        b_var = b.variable
        # not calling patternbroadcast if not broadcastable, because it seems
        # to have a small overhead
        if any(broadcastable):
            b_var = T.patternbroadcast(b_var, broadcastable)
        network.create_vw(
            name="default",
            variable=(in_vw.variable + b_var),
            shape=in_vw.shape,
            tags={"output"},
        )


@core.register_node("linear_mapping")
class LinearMappingNode(core.NodeImpl):

    """
    node that applies a linear mapping to the last dimension of its input
    (a dot product with a parameter)
    """

    hyperparameter_names = ("linear_mapping_inits",
                            "inits",
                            "output_dim")

    def compute_output(self, network, in_vw):
        output_dim = network.find_hyperparameter(["output_dim"])
        weight_shape = (in_vw.shape[-1], output_dim)
        output_shape = tuple(in_vw.shape[:-1]) + (output_dim, )
        W = network.create_vw(
            name="weight",
            is_shared=True,
            shape=weight_shape,
            tags={"parameter", "weight"},
            default_inits=[],
            default_inits_hyperparameters=["linear_mapping_inits", "inits"]
        )
        network.create_vw(
            name="default",
            variable=T.dot(in_vw.variable, W.variable),
            shape=output_shape,
            tags={"output"},
        )


@core.register_node("apply")
class ApplyNode(core.NodeImpl):

    """
    applies a pure theano function
    """

    hyperparameter_names = ("fn", "shape_fn")

    def compute_output(self, network, in_vw):
        fn = network.find_hyperparameter(["fn"])
        shape_fn = network.find_hyperparameter(["shape_fn"], None)
        if shape_fn is None:
            shape = None
        else:
            shape = shape_fn(in_vw.shape)
        network.create_vw(
            name="default",
            variable=fn(in_vw.variable),
            shape=shape,
            tags={"output"},
        )


@core.register_node("add_constant")
class AddConstantNode(core.NodeImpl):

    """
    adds a constant value to its input
    """

    hyperparameter_names = ("value", )

    def compute_output(self, network, in_vw):
        value = network.find_hyperparameter(["value"])
        network.create_vw(
            name="default",
            variable=in_vw.variable + value,
            shape=in_vw.shape,
            tags={"output"},
        )


@core.register_node("multiply_constant")
class MultiplyConstantNode(core.NodeImpl):

    """
    adds a constant value to its input
    """

    hyperparameter_names = ("value", )

    def compute_output(self, network, in_vw):
        value = network.find_hyperparameter(["value"])
        network.create_vw(
            name="default",
            variable=in_vw.variable * value,
            shape=in_vw.shape,
            tags={"output"},
        )
