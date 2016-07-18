import itertools

import numpy as np
import theano
import theano.tensor as T

# for importing
from ..core.inits import (SharedInit,
                          WeightInit,
                          LinearWeightInit,
                          ConstantInit,
                          ZeroInit,
                          PreallocatedInit)


# ################################ constants ################################


# NOTE: these may be wrong
# ---
# eg. for a conv weight in theano, first axis is output filters and second
# axis is number of input channels but for a linear mapping node, the first
# axis is the input and the second is the output
# ---
# preferring defaults for convs, because linear mappings are symmetric
DEFAULT_IN_AXES = (1,)
DEFAULT_OUT_AXES = (0,)


def leaky_relu_gain(leak_alpha):
    """
    from https://github.com/Lasagne/Lasagne/issues/468
    """
    return np.sqrt(2 / (1 + leak_alpha ** 2))

# sane defaults for gains
GAINS = dict(
    relu=leaky_relu_gain(0),
    leaky_relu=leaky_relu_gain(0.01),
    very_leaky_relu=leaky_relu_gain(1 / 3.),
    tanh=1.1,
    linear=1,
)

# ############################## special inits ##############################


class TiedInit(SharedInit):

    """
    uses already defined shared variables for node with name `root_node_name`
    from node with name `target_node_name`
    """

    def __init__(self, root_node_name, target_root_node_name):
        self.root_node_name = root_node_name
        self.target_root_node_name = target_root_node_name

    def create_shared(self, vw):
        network = vw.relative_network
        assert network is not None
        node_name = network._name
        assert self.root_node_name in node_name
        target_node_name = node_name.replace(self.root_node_name,
                                             self.target_root_node_name,
                                             1)
        # HACK there should be a better way to to do this!
        node_name2, vw_key = vw.name.split(":")
        # sanity check, just to be sure
        assert node_name2 == node_name
        shared = network[target_node_name].get_vw(vw_key).variable
        assert shared.dtype == vw.dtype
        assert shared.get_value().shape == vw.shape
        assert shared.broadcastable == vw.broadcastable
        # mutate variable wrapper to no longer be shared
        vw.is_shared_ = False
        vw.tags_ = {"tied"}
        return theano.compile.view_op(shared)


# ############################### weight inits ###############################


class NormalWeightInit(WeightInit):

    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def initialize_value(self, vw):
        return np.random.normal(loc=self.mean,
                                scale=self.std,
                                size=vw.shape)


class UniformWeightInit(WeightInit):

    def __init__(self, range_=0.01):
        try:
            # range_ is a tuple
            low, high = range_
        except TypeError:
            # range is a scalar
            low, high = -range_, range_
        self.low = low
        self.high = high

    def initialize_value(self, vw):
        return np.random.uniform(low=self.low,
                                 high=self.high,
                                 size=vw.shape)


def xavier_magnitude(shape, in_axes, out_axes, gain):
    """
    NOTE: does not differentiate between in_axes and out_axes, so they
    can be switched
    """
    shape = np.array(shape)
    other_axes_size = np.prod([s
                               for dim, s in enumerate(shape)
                               if not ((dim in in_axes) or
                                       (dim in out_axes))])
    in_axes_size = np.prod(shape[in_axes])
    out_axes_size = np.prod(shape[out_axes])

    base = np.sqrt(2.0 / ((in_axes_size + out_axes_size) * other_axes_size))
    return base * gain


class XavierNormalInit(LinearWeightInit):

    def __init__(self,
                 gain=1,
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        self.gain = gain
        self.in_axes = in_axes
        self.out_axes = out_axes

    def initialize_value(self, vw):
        magnitude = xavier_magnitude(vw.shape,
                                     in_axes=self.in_axes,
                                     out_axes=self.out_axes,
                                     gain=self.gain)
        return np.random.normal(loc=0,
                                scale=magnitude,
                                size=vw.shape)


class XavierUniformInit(LinearWeightInit):

    def __init__(self,
                 gain=1,
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        self.gain = gain
        self.in_axes = in_axes
        self.out_axes = out_axes

    def predicate(self, vw):
        return super(XavierUniformInit, self).predicate(vw) and vw.ndim >= 2

    def initialize_value(self, vw):
        magnitude = np.sqrt(3) * xavier_magnitude(vw.shape,
                                                  in_axes=self.in_axes,
                                                  out_axes=self.out_axes,
                                                  gain=self.gain)
        return np.random.uniform(low=-magnitude,
                                 high=magnitude,
                                 size=vw.shape)


def he_magnitude(shape, in_axes, out_axes, gain):
    """
    http://arxiv.org/abs/1502.01852

    NOTE: also called MSR init

    NOTE: does not differentiate between in_axes and out_axes, so they
    can be switched
    """
    # consider all non-out_axes as in_axes
    in_axes_size = np.prod([s
                            for dim, s in enumerate(shape)
                            if dim not in out_axes])
    # NOTE: this is actually sqrt(2) in the paper, but a gain of sqrt(2)
    # is recommended for ReLUs
    base = np.sqrt(1.0 / in_axes_size)
    return base * gain


class HeNormalInit(LinearWeightInit):

    def __init__(self,
                 gain=1,
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        self.gain = gain
        self.in_axes = in_axes
        self.out_axes = out_axes

    def predicate(self, vw):
        return super(HeNormalInit, self).predicate(vw) and vw.ndim >= 2

    def initialize_value(self, vw):
        magnitude = he_magnitude(vw.shape,
                                 in_axes=self.in_axes,
                                 out_axes=self.out_axes,
                                 gain=self.gain)
        return np.random.normal(loc=0,
                                scale=magnitude,
                                size=vw.shape)


class HeUniformInit(LinearWeightInit):

    def __init__(self,
                 gain=1,
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        self.gain = gain
        self.in_axes = in_axes
        self.out_axes = out_axes

    def initialize_value(self, vw):
        magnitude = np.sqrt(3) * he_magnitude(vw.shape,
                                              in_axes=self.in_axes,
                                              out_axes=self.out_axes,
                                              gain=self.gain)
        return np.random.uniform(low=-magnitude,
                                 high=magnitude,
                                 size=vw.shape)


class OrthogonalInit(LinearWeightInit):

    """
    http://arxiv.org/abs/1312.6120
    """

    def __init__(self,
                 gain=1,
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        self.gain = gain
        self.in_axes = in_axes
        self.out_axes = out_axes

    def initialize_value(self, vw):
        shape = vw.shape
        assert len(shape) >= 2
        # TODO can get around this by making all output dimensions
        # side-by-side (eg. (0, 1, 2))
        assert len(self.out_axes) == 1
        # consider all non-out_axes as in_axes
        in_axes_size = np.prod([s
                                for dim, s in enumerate(shape)
                                if dim not in self.out_axes])
        out_axes_size = shape[self.out_axes[0]]
        # using logic similar to lasagne's
        flat_shape = (out_axes_size, in_axes_size)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return self.gain * q.reshape(shape)


class SparseInit(LinearWeightInit):

    def __init__(self, ratio, weights_init, sparse_axes, sparse_init=None):
        """
        original motivation: hard limit the number of non-zero incoming
        connection weights to each unit

        from "Deep learning via Hessian-free optimization" 2010

        ratio: fraction of non-sparse values along the sparse axis

        weights_init: initialization scheme to use for non-sparse entries

        sparse_axes: axes along which each has the exact amount of sparsity
        eg. if sparse_axes == (1,), and W in R^2, then
        (W != 0).mean(axis=0) == ratio (axis 1 has exactly ratio non-sparse)

        sparse_init: initialization scheme to use for sparse entries
        (default: ZeroInit())
        """
        if sparse_init is None:
            sparse_init = ZeroInit()
        self.ratio = ratio
        self.weights_init = weights_init
        self.sparse_axes = sparse_axes
        self.sparse_init = sparse_init

    def initialize_value(self, vw):
        shape = vw.shape
        res = self.sparse_init.initialize_value(vw)
        non_sparse = self.weights_init.initialize_value(vw)

        mask_shape = [s
                      for idx, s in enumerate(shape)
                      if idx not in self.sparse_axes]
        tmp_idx = [slice(None) for _ in shape]
        for idxs in itertools.product(*[list(range(shape[axis]))
                                        for axis in self.sparse_axes]):
            # construct an index into res
            # eg. (:, :, 3, :, 4)
            for idx, axis in zip(idxs, self.sparse_axes):
                tmp_idx[axis] = idx
            res_idx = tuple(tmp_idx)

            mask_vals = np.random.uniform(size=mask_shape)
            # NOTE: ratio = fraction of NON sparse values
            # ratio = 1 -> not sparse at all
            # ratio = 0 -> 100% sparse
            mask = mask_vals <= np.percentile(mask_vals, 100 * self.ratio)
            res[res_idx][mask] = non_sparse[res_idx][mask]

        # TODO should outputs be rescaled by 1 / self.ratio
        # (to keep variance between layers ~1)
        return res


class RandomWalkInit(LinearWeightInit):

    """
    from
    "Random Walk Initialization for Training Very Deep Feedforward Networks"
    http://arxiv.org/abs/1412.6558
    """

    def __init__(self,
                 activation="relu",
                 in_axes=DEFAULT_IN_AXES,
                 out_axes=DEFAULT_OUT_AXES):
        assert activation in {"relu", "linear"}
        self.activation = activation
        self.in_axes = in_axes
        self.out_axes = out_axes

    def initialize_value(self, vw):
        N = np.prod([s
                     for dim, s in enumerate(vw.shape)
                     if dim in self.out_axes])
        if self.activation == "linear":
            g = np.exp(1. / (2 * N))
        elif self.activation == "relu":
            g = np.sqrt(2) * np.exp(1.2 / (max(N, 6) - 2.4))
        magnitude = g / N
        return np.random.normal(loc=0,
                                scale=magnitude,
                                size=vw.shape)
