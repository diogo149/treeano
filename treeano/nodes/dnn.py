"""
nodes based on cuDNN
http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
"""

from theano.sandbox.cuda import dnn

from .. import core

from . import downsample


@core.register_node("dnn_pool")
class DnnPoolNode(core.NodeImpl):

    """
    2D or 3D pooling node that takes in a specified "mode" and utilizes cuDNN
    """

    hyperparameter_names = ("mode",
                            "pool_size",
                            "pool_stride",
                            "stride",
                            "pool_pad",
                            "pad")

    def compute_output(self, network, in_vw):
        mode = network.find_hyperparameter(["mode"])
        pool_size = network.find_hyperparameter(["pool_size"])
        dim = len(pool_size)
        # works for sizes 2 and 3
        assert dim in [2, 3]
        stride = network.find_hyperparameter(["pool_stride",
                                              "stride"],
                                             (1,) * dim)
        pad = network.find_hyperparameter(["pool_pad", "pad"], (0,) * dim)
        assert dim == len(stride) == len(pad)
        if dim == 2:
            pool_axes = (2, 3)
        elif dim == 3:
            pool_axes = (2, 3, 4)
        out_shape = downsample.pool_output_shape(
            input_shape=in_vw.shape,
            axes=pool_axes,
            pool_shape=pool_size,
            strides=stride,
            pads=pad)
        out_var = dnn.dnn_pool(img=in_vw.variable,
                               ws=pool_size,
                               stride=stride,
                               pad=pad,
                               mode=mode)

        network.create_variable(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


def DnnMeanPoolNode(*args, **kwargs):
    """
    NOTE: this average does not include padding
    """
    return DnnPoolNode(*args, mode="average_exc_pad", **kwargs)


def DnnMaxPoolNode(*args, **kwargs):
    return DnnPoolNode(*args, mode="max", **kwargs)
