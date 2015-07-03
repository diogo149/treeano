import theano
import theano.tensor as T

from .. import core
from .. import utils


@core.register_node("feature_pool")
class FeaturePoolNode(core.NodeImpl):

    hyperparameter_names = ("num_pieces",
                            "feature_pool_axis",
                            "axis",
                            "pool_function")

    def compute_output(self, network, in_vw):
        k = network.find_hyperparameter(["num_pieces"])
        pool_fn = network.find_hyperparameter(["pool_function"])
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

        # output calculation
        in_var = in_vw.variable
        symbolic_shape = in_vw.symbolic_shape()
        new_symbolic_shape = (symbolic_shape[:axis]
                              + (out_shape[axis], k) +
                              symbolic_shape[axis + 1:])
        reshaped = in_var.reshape(new_symbolic_shape)
        out_var = pool_fn(reshaped, axis=axis + 1)

        network.create_variable(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("maxout")
class MaxoutNode(core.Wrapper0NodeImpl):

    """
    from "Maxout Networks" http://arxiv.org/abs/1302.4389
    """
    hyperparameter_names = tuple(filter(lambda x: x != "pool_function",
                                        FeaturePoolNode.hyperparameter_names))

    def architecture_children(self):
        return [FeaturePoolNode(self.name + "_featurepool")]

    def get_hyperparameter(self, network, name):
        if name == "pool_function":
            return T.max
        else:
            return super(MaxoutNode, self).get_hyperparameter(network, name)
