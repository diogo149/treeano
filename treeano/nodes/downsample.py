import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from .. import core
from .. import utils


def pool_output_length(input_size,
                       pool_size,
                       stride,
                       pad,
                       ignore_border):
    """
    calculates the output size along a single axis for a pooling operation

    logic from theano.tensor.signal.pool.Pool.out_shape
    """
    if input_size is None:
        return None

    if ignore_border:
        without_stride = input_size + 2 * pad - pool_size + 1
        # equivalent to np.ceil(without_stride / stride)
        pre_max = (without_stride + stride - 1) // stride
        output_size = utils.maximum(pre_max, 0)
    else:
        assert pad == 0
        if stride >= pool_size:
            output_size = (input_size + stride - 1) // stride
        else:
            pre_max = (input_size - pool_size + stride - 1) // stride
            output_size = 1 + utils.maximum(0, pre_max)

    return output_size


def pool_output_shape(input_shape,
                      axes,
                      pool_shape,
                      strides,
                      pads,
                      ignore_border=True):
    """
    compute output shape for a pool
    """
    if strides is None:
        strides = pool_shape

    output_shape = list(input_shape)
    for axis, pool_size, stride, pad in zip(axes,
                                            pool_shape,
                                            strides,
                                            pads):
        output_shape[axis] = pool_output_length(input_shape[axis],
                                                pool_size,
                                                stride,
                                                pad,
                                                ignore_border)
    return tuple(output_shape)


def pool_output_shape_2d(input_shape,
                         axes,
                         pool_shape,
                         strides,
                         pads,
                         ignore_border=True):
    """
    compute output shape for a pool
    """
    return tuple(DownsampleFactorMax.out_shape(
        imgshape=input_shape,
        ds=pool_shape,
        st=strides,
        ignore_border=ignore_border,
        padding=pads,
    ))


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

        network.create_vw(
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
    hyperparameter_names = tuple(
        [n for n in FeaturePoolNode.hyperparameter_names
         if n != "pool_function"])

    def architecture_children(self):
        return [FeaturePoolNode(self.name + "_featurepool")]

    def init_state(self, network):
        super(MaxoutNode, self).init_state(network)
        network.set_hyperparameter(self.name + "_featurepool",
                                   "pool_function",
                                   T.max)


@core.register_node("pool_2d")
class Pool2DNode(core.NodeImpl):

    """
    2D pooling node that takes in a specified "mode"
    """

    hyperparameter_names = ("mode",
                            "pool_size",
                            "pool_stride",
                            "stride",
                            "pool_pad",
                            "pad",
                            "ignore_border")

    def compute_output(self, network, in_vw):
        mode = network.find_hyperparameter(["mode"])
        pool_size = network.find_hyperparameter(["pool_size"])
        stride = network.find_hyperparameter(["pool_stride",
                                              "stride"],
                                             None)
        pad = network.find_hyperparameter(["pool_pad", "pad"], (0, 0))
        ignore_border = network.find_hyperparameter(["ignore_border"],
                                                    True)
        if ((stride is not None)
                and (stride != pool_size)
                and (not ignore_border)):
            # as of 20150813
            # for more information, see:
            # https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
            msg = ("Setting stride not equal to pool size and not ignoring"
                   " border results in using a slower (cpu-based)"
                   " implementation")
            # making this an assertion instead of a warning to make sure it
            # is done
            assert False, msg

        out_shape = pool_output_shape(
            input_shape=in_vw.shape,
            axes=(2, 3),
            pool_shape=pool_size,
            strides=stride,
            pads=pad)
        out_var = max_pool_2d(input=in_vw.variable,
                              ds=pool_size,
                              st=stride,
                              ignore_border=ignore_border,
                              padding=pad,
                              mode=mode)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


def MeanPool2DNode(*args, **kwargs):
    """
    NOTE: this average does not include padding
    """
    return Pool2DNode(*args, mode="average_exc_pad", **kwargs)


def MaxPool2DNode(*args, **kwargs):
    return Pool2DNode(*args, mode="max", **kwargs)


def SumPool2DNode(*args, **kwargs):
    """
    NOTE: does not work on the GPU, and may be slower than mean pool
    """
    return Pool2DNode(*args, mode="sum", **kwargs)


@core.register_node("global_pool_2d")
class GlobalPool2DNode(core.NodeImpl):

    """
    pools all 2D spatial locations into a single value with a given "mode"
    """

    hyperparameter_names = ("mode",)

    def compute_output(self, network, in_vw):
        mode = network.find_hyperparameter(["mode"])
        out_shape = in_vw.shape[:2]
        pool_size = in_vw.shape[2:]
        pooled = max_pool_2d(in_vw.variable,
                             ds=pool_size,
                             mode=mode,
                             # doesn't make a different here,
                             # but allows using cuDNN
                             ignore_border=True)
        out_var = pooled.flatten(2)
        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


def GlobalMeanPool2DNode(*args, **kwargs):
    """
    NOTE: this average does not include padding
    """
    return GlobalPool2DNode(*args, mode="average_exc_pad", **kwargs)


def GlobalMaxPool2DNode(*args, **kwargs):
    return GlobalPool2DNode(*args, mode="max", **kwargs)


def GlobalSumPool2DNode(*args, **kwargs):
    return GlobalPool2DNode(*args, mode="sum", **kwargs)


@core.register_node("custom_pool_2d")
class CustomPool2DNode(core.NodeImpl):

    """
    2D pooling node that allows providing a custom pool function
    """

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
            pool_shape=pool_size,
            strides=stride,
            pads=pads,
        )
        out_shape = pool_output_shape(
            input_shape=in_vw.shape,
            **shape_kwargs)
        symbolic_out_shape = pool_output_shape(
            input_shape=in_vw.symbolic_shape(),
            **shape_kwargs)

        # compute output
        neibs = images2neibs(ten4=in_vw.variable,
                             neib_shape=pool_size,
                             neib_step=stride,
                             mode=pool_mode)
        feats = pool_fn(neibs, axis=1)
        out_var = feats.reshape(symbolic_out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@core.register_node("custom_global_pool")
class CustomGlobalPoolNode(core.NodeImpl):

    """
    pools all spatial locations into a single value with a custom pool function
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
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape[:2],
            tags={"output"},
        )
