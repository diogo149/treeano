import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

from .. import core
from .. import utils


@core.register_node("feature_pool")
class FeaturePoolNode(core.NodeImpl):

    hyperparameter_names = ("num_pieces",
                            "feature_pool_axis",
                            "axis",
                            "pool_function")

    def compute_output(self, network, in_vw):
        k = network.find_hyperparameter(["num_pieces"], 2)
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
    hyperparameter_names = filter(lambda x: x != "pool_function",
                                  FeaturePoolNode.hyperparameter_names)

    def architecture_children(self):
        return [FeaturePoolNode(self.name + "_featurepool")]

    def init_state(self, network):
        super(MaxoutNode, self).init_state(network)
        network.set_hyperparameter(self.name + "_featurepool",
                                   "pool_function",
                                   T.max)


@core.register_node("pool_2d")
class Pool2DNode(core.NodeImpl):

    hyperparameter_names = ("pool_function",
                            "pool_size",
                            "pool_stride",
                            "stride")

    def compute_output(self, network, in_vw):
        # hyperparameters
        pool_fn = network.find_hyperparameter(["pool_function"])
        pool_size = network.find_hyperparameter(["pool_size"])
        stride = network.find_hyperparameter(["pool_stride",
                                              "stride"],
                                             pool_size)
        # TODO parameterize
        # ---
        # TODO assumes pooling across axis 2 and 3
        pooling_axes = (2, 3)
        # maybe have a bool (ignore_borders=False) instead of a string
        pool_mode = "valid"
        pads = (0, 0)

        # calculate shapes
        shape_kwargs = dict(
            axes=pooling_axes,
            local_sizes=pool_size,
            strides=stride,
            pads=pads,
        )
        out_shape = utils.local_computation_output_shape(
            input_shape=in_vw.shape, **shape_kwargs)
        symbolic_out_shape = utils.local_computation_output_shape(
            input_shape=in_vw.symbolic_shape(), **shape_kwargs)

        # compute output
        neibs = images2neibs(ten4=in_vw.variable,
                             neib_shape=pool_size,
                             neib_step=stride,
                             mode=pool_mode)
        feats = pool_fn(neibs, axis=1)
        out_var = feats.reshape(symbolic_out_shape)

        network.create_variable(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("mean_pool_2d")
class MeanPool2DNode(core.Wrapper0NodeImpl):

    hyperparameter_names = filter(lambda x: x != "pool_function",
                                  Pool2DNode.hyperparameter_names)

    def architecture_children(self):
        return [Pool2DNode(self.name + "_pool2d")]

    def init_state(self, network):
        super(MeanPool2DNode, self).init_state(network)
        network.set_hyperparameter(self.name + "_pool2d",
                                   "pool_function",
                                   T.mean)


@core.register_node("global_pool")
class GlobalPoolNode(core.NodeImpl):

    """
    pools all spatial locations into a single value
    """

    hyperparameter_names = ("pool_function",)

    def compute_output(self, network, in_vw):
        pool_fn = network.find_hyperparameter(["pool_function"])
        # FIXME generalize to other axes
        # assume that spatial locations are all trailing locations
        # 3-tensor with all spatial axes flattened into the final one:
        flattened = in_vw.variable.flatten(3)
        # pool together
        out_var = pool_fn(flattened, axis=2)
        network.create_variable(
            "default",
            variable=out_var,
            shape=in_vw.shape[:2],
            tags={"output"},
        )
