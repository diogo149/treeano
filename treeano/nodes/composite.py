"""
nodes which are combinations of multiple other nodes
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

from .. import core
from . import simple
from . import containers


def flatten_2d(v):
    if v.ndim > 2:
        return T.flatten(v, outdim=2)
    elif v.ndim == 2:
        return v
    else:
        raise ValueError


def flatten_2d_shape(shape):
    if len(shape) > 2:
        return (shape[0], np.prod(shape[1:]))
    elif len(shape) == 2:
        return shape
    else:
        raise ValueError


@core.register_node("dense")
class DenseNode(core.Wrapper1NodeImpl):

    """
    applies a dense neural network layer to the input

    takes in a single node to be used as the activation function
    """

    hyperparameter_names = ("num_units",
                            "shared_initializations",
                            "initializations",
                            "inits")

    def architecture_children(self):
        children = super(DenseNode, self).architecture_children()
        activation_node, = children
        return [
            containers.SequentialNode(
                self._name + "_sequential",
                [simple.ApplyNode(self._name + "_flatten",
                                  fn=flatten_2d,
                                  shape_fn=flatten_2d_shape),
                 simple.LinearMappingNode(self._name + "_linear"),
                 simple.AddBiasNode(self._name + "_bias"),
                 activation_node])]

    def get_hyperparameter(self, network, name):
        if name == "output_dim":
            # remap a child looking for "output_dim" to "num_units"
            return network.find_hyperparameter(["num_units"])
        else:
            return super(DenseNode, self).get_hyperparameter(network, name)
