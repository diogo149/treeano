"""
a bunch of simple, composable nodes - not necessarily simple in their
implementation, but simple in that they do a single thing
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T

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

    def __init__(self, name, node, **hyperparameters):
        # set hyperparameter keys to be all passed in keys
        self.hyperparameter_names = hyperparameters.keys()
        # override init to allow for using keyword arguments
        super(HyperparameterNode, self).__init__(name, node, **hyperparameters)


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
        network.create_variable(
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


@core.register_node("fn_combine")
class FunctionCombineNode(core.NodeImpl):

    """
    Combines each of its inputs with the given combine function

    NOTE: inputs are passed in sorted according to their to_key

    combine_fn:
    function that takes in several theano variables and returns a new
    theano variable

    shape_fn:
    optional function to calculate the shape of the output given the shapes
    of the inputs
    """

    hyperparameter_names = ("combine_fn",
                            "shape_fn")

    def init_state(self, network):
        self.input_keys = tuple(sorted(network.get_all_input_edges().keys()))

    def compute_output(self, network, *input_vws):
        combine_fn = network.find_hyperparameter(["combine_fn"])
        shape_fn = network.find_hyperparameter(["shape_fn"], None)
        if shape_fn is None:
            shape = None
        else:
            shape = shape_fn(*[input_vw.shape for input_vw in input_vws])
        var = combine_fn(*[input_vw.variable for input_vw in input_vws])
        network.create_variable(
            name="default",
            variable=var,
            shape=shape,
            tags={"output"}
        )


@core.register_node("constant")
class ConstantNode(core.NodeImpl):

    """
    node that returns a constant value
    """

    hyperparameter_names = ("constant_value", "value")
    input_keys = ()

    def compute_output(self, network):
        value = network.find_hyperparameter(["constant_value", "value"])
        shape = value.shape if hasattr(value, "shape") else ()
        network.create_variable(
            name="default",
            variable=T.constant(value),
            shape=shape,
            tags={"output"},
        )


@core.register_node("add_bias")
class AddBiasNode(core.NodeImpl):

    """
    node that adds a bias parameter to its input
    """

    hyperparameter_names = ("shared_initializations",
                            "initializations",
                            "inits",
                            "broadcastable")

    def compute_output(self, network, in_var):
        inits = network.find_hyperparameter(["shared_initializations",
                                             "initializations",
                                             "inits"],
                                            None)
        broadcastable = network.find_hyperparameter(
            ["broadcastable"],
            (False,) * in_var.ndim)
        assert len(broadcastable) == in_var.ndim
        shape = tuple([1 if is_broadcastable else size
                       for is_broadcastable, size in zip(broadcastable,
                                                         in_var.shape)])
        b = network.create_variable(
            name="bias",
            is_shared=True,
            shape=shape,
            tags={"parameter", "bias"},
            shared_initializations=inits,
        )
        network.create_variable(
            name="default",
            variable=(in_var.variable + b.variable),
            shape=in_var.shape,
            tags={"output"},
        )


@core.register_node("linear_mapping")
class LinearMappingNode(core.NodeImpl):

    """
    node that applies a linear mapping to the last dimension of its input
    (a dot product with a parameter)
    """

    hyperparameter_names = ("shared_initializations",
                            "initializations",
                            "inits",
                            "output_dim")

    def compute_output(self, network, in_var):
        inits = network.find_hyperparameter(["shared_initializations",
                                             "initializations",
                                             "inits"],
                                            None)
        output_dim = network.find_hyperparameter(["output_dim"])
        weight_shape = (in_var.shape[-1], output_dim)
        output_shape = tuple(in_var.shape[:-1]) + (output_dim, )
        W = network.create_variable(
            name="weight",
            is_shared=True,
            shape=weight_shape,
            tags={"parameter", "weight"},
            shared_initializations=inits,
        )
        network.create_variable(
            name="default",
            variable=T.dot(in_var.variable, W.variable),
            shape=output_shape,
            tags={"output"},
        )


@core.register_node("apply")
class ApplyNode(core.NodeImpl):

    """
    applies a pure theano function
    """

    hyperparameter_names = ("fn", "shape_fn")

    def compute_output(self, network, in_var):
        fn = network.find_hyperparameter(["fn"])
        shape_fn = network.find_hyperparameter(["shape_fn"], None)
        if shape_fn is None:
            shape = None
        else:
            shape = shape_fn(in_var.shape)
        network.create_variable(
            name="default",
            variable=fn(in_var.variable),
            shape=shape,
            tags={"output"},
        )
