"""
nodes for combining the outputs of multiple nodes
"""

import abc
import six

import theano.tensor as T

from .. import utils
from .. import core


class BaseCombineNode(six.with_metaclass(abc.ABCMeta, core.WrapperNodeImpl)):

    """
    base node class for combining multiple outputs together
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


@core.register_node("concatenate")
class ConcatenateNode(BaseCombineNode):

    """
    like theano.tensor.concatenate
    """

    hyperparameter_names = ("concatenate_axis",
                            "axis")

    def compute_output(self, network, *in_vws):
        # find axis
        axis = network.find_hyperparameter(["concatenate_axis",
                                            "axis"],
                                           None)
        if axis is None:
            batch_axis = network.find_hyperparameter(["batch_axis"])
            if batch_axis is None:
                # by default, be the first axis
                axis = 0
            else:
                # by default, be the first non-batch axis
                axis = 0 if batch_axis == 0 else 1

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

        network.create_variable(
            "default",
            variable=T.concatenate([vw.variable for vw in in_vws],
                                   axis),
            shape=tuple(shape),
            tags={"output"},
        )


# class ScalarSumNode(BaseCombineNode):

#     """
#     computes a sum of the outputs of the inputs
#     """
#     pass
