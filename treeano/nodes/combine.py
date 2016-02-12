"""
nodes for combining the outputs of multiple nodes
"""

import abc

import six
import theano.tensor as T

from .. import utils
from .. import core


# ############################### base classes ###############################


class BaseChildrenCombineNode(six.with_metaclass(abc.ABCMeta,
                                                 core.WrapperNodeImpl)):

    """
    base node class for combining the outputs of a node's children together
    """

    @property
    def input_keys(self):
        return ["child%d" % idx
                for idx in range(len(self.architecture_children()))]

    def init_state(self, network):
        """
        forward the input of this node to each of the children as a default
        """
        children = self.architecture_children()
        assert (len(children) == len(self.input_keys))
        for to_key, child in zip(self.input_keys, children):
            network.forward_input_to(child.name)
            network.take_output_from(child.name, to_key=to_key)

    @abc.abstractmethod
    def compute_output(self, network, *in_vws):
        pass


class BaseInputCombineNode(six.with_metaclass(abc.ABCMeta, core.NodeImpl)):

    """
    base node class for combining all inputs of a node together

    example use case: having a sum node that combines all inputs from
    SendToNode's (eg. a main cost node where all other costs are sent to it)
    """

    hyperparameter_names = ("ignore_default_input",)

    def init_state(self, network):
        """
        forward the input of this node to each of the children as a default
        """
        ignore_default_input = network.find_hyperparameter(
            ["ignore_default_input"], False)
        self.input_keys = tuple(sorted(network.get_all_input_edges().keys()))
        if ignore_default_input:
            self.input_keys = tuple([k for k in self.input_keys
                                     if k != "default"])

    @abc.abstractmethod
    def compute_output(self, network, *in_vws):
        pass


# ############################# implementations #############################


@core.register_node("input_fn_combine")
class InputFunctionCombineNode(BaseInputCombineNode):

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

    def compute_output(self, network, *in_vws):
        combine_fn = network.find_hyperparameter(["combine_fn"])
        shape_fn = network.find_hyperparameter(["shape_fn"], None)
        if shape_fn is None:
            shape = None
        else:
            shape = shape_fn(*[input_vw.shape for input_vw in in_vws])
        var = combine_fn(*[input_vw.variable for input_vw in in_vws])
        network.create_vw(
            name="default",
            variable=var,
            shape=shape,
            tags={"output"}
        )


@core.register_node("concatenate")
class ConcatenateNode(BaseChildrenCombineNode):

    """
    like theano.tensor.concatenate
    """

    hyperparameter_names = ("concatenate_axis",
                            "axis")

    def compute_output(self, network, *in_vws):
        # find axis
        axis = network.find_hyperparameter(
            ["concatenate_axis",
             "axis"],
            # by default, the first non-batch axis
            utils.nth_non_batch_axis(network, 0))

        # calculate shape
        input_shapes = [vw.shape for vw in in_vws]
        assert utils.all_equal(map(len, input_shapes)), dict(
            msg="all inputs must have the same shape",
            input_shapes=input_shapes,
        )
        assert axis <= len(input_shapes[0])
        shape = []
        for idx, sizes in enumerate(zip(*input_shapes)):
            if idx == axis:
                if any(s is None for s in sizes):
                    shape.append(None)
                else:
                    shape.append(sum(sizes))
            else:
                assert utils.all_equal(sizes), dict(
                    msg=("all sizes on the axis not being concatenated must "
                         "be equal"),
                    input_shapes=input_shapes,
                    axis=idx,
                )
                shape.append(sizes[0])

        network.create_vw(
            "default",
            variable=T.concatenate([vw.variable for vw in in_vws],
                                   axis),
            shape=tuple(shape),
            tags={"output"},
        )


def elementwise_sum(network, *in_vws):
    # calculate and verify shape
    shape = utils.vw_reduce_shape(in_vws)
    network.create_vw(
        "default",
        variable=utils.smart_sum([vw.variable for vw in in_vws]),
        shape=shape,
        tags={"output"},
    )


@core.register_node("elementwise_sum")
class ElementwiseSumNode(BaseChildrenCombineNode):

    """
    computes a sum of the outputs of the node's children
    """

    def compute_output(self, network, *in_vws):
        elementwise_sum(network, *in_vws)


@core.register_node("input_elementwise_sum")
class InputElementwiseSumNode(BaseInputCombineNode):

    """
    computes a sum of the inputs of the node
    """

    def compute_output(self, network, *in_vws):
        elementwise_sum(network, *in_vws)


def elementwise_product(network, *in_vws):
    # calculate and verify shape
    shape = utils.vw_reduce_shape(in_vws)
    network.create_vw(
        "default",
        variable=utils.smart_product([vw.variable for vw in in_vws]),
        shape=shape,
        tags={"output"},
    )


@core.register_node("elementwise_product")
class ElementwiseProductNode(BaseChildrenCombineNode):

    """
    computes a product of the outputs of the node's children
    """

    def compute_output(self, network, *in_vws):
        elementwise_product(network, *in_vws)
