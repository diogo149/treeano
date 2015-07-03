import abc
import six

import theano.tensor as T

from .. import core
from .. import utils


class BaseActivationNode(six.with_metaclass(abc.ABCMeta, core.NodeImpl)):

    """
    base node class for activation functions
    """

    def compute_output(self, network, in_vw):
        new_var = self.activation(network, in_vw)
        network.create_variable(
            "default",
            variable=new_var,
            shape=in_vw.shape,
            tags={"output"},
        )

    @abc.abstractmethod
    def activation(self, network, in_vw):
        pass


def relu(x):
    return 0.5 * (x + abs(x))


@core.register_node("relu")
class ReLUNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return relu(in_vw.variable)


@core.register_node("softmax")
class SoftmaxNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.nnet.softmax(in_vw.variable)


@core.register_node("tanh")
class TanhNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.tanh(in_vw.variable)


@core.register_node("scaled_tanh")
class ScaledTanhNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return 1.7159 * T.tanh(in_vw.variable * (2.0 / 3.0))


@core.register_node("sigmoid")
class SigmoidNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return T.nnet.sigmoid(in_vw.variable)


@core.register_node("resqrt")
class ReSQRTNode(BaseActivationNode):

    """
    rectified shifted square root
    from "Author Identification using Multi-headed Recurrent Neural Networks"
    http://arxiv.org/abs/1506.04891
    """

    def activation(self, network, in_vw):
        r = relu(in_vw.variable)
        return T.sqrt(r + 1) - 1


@core.register_node("abs")
class AbsNode(BaseActivationNode):

    def activation(self, network, in_vw):
        return abs(in_vw.variable)


@core.register_node("channel_out")
class ChannelOutNode(BaseActivationNode):

    """
    from "From Maxout to Channel-Out: Encoding Information on Sparse Pathways"
    http://arxiv.org/abs/1312.1909
    """

    hyperparameter_names = ("num_pieces",
                            "feature_pool_axis",
                            "axis")

    def activation(self, network, in_vw):
        # NOTE: mostly copied from FeaturePoolNode
        k = network.find_hyperparameter(["num_pieces"])
        axis = network.find_hyperparameter(
            ["feature_pool_axis",
             "axis"],
            # by default, the first non-batch axis
            utils.nth_non_batch_axis(network, 0))

        # shape calculation
        in_shape = in_vw.shape
        in_features = in_shape[axis]
        assert (in_features % k) == 0
        out_shape = list(in_shape)
        out_shape[axis] = in_shape[axis] // k
        out_shape = tuple(out_shape)

        # calculate indices of maximum activation
        in_var = in_vw.variable
        symbolic_shape = in_vw.symbolic_shape()
        new_symbolic_shape = (symbolic_shape[:axis]
                              + (out_shape[axis], k) +
                              symbolic_shape[axis + 1:])
        reshaped = in_var.reshape(new_symbolic_shape)
        max_idxs = T.argmax(reshaped, axis=axis + 1, keepdims=True)

        # calculate indices of each unit
        arange_pattern = ["x"] * (in_vw.ndim + 1)
        arange_pattern[axis + 1] = 0
        idxs = T.arange(k).dimshuffle(tuple(arange_pattern))

        mask = T.eq(max_idxs, idxs).reshape(symbolic_shape)
        return in_vw.variable * mask
