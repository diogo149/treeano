import numpy as np
import theano
import theano.tensor as T

fX = theano.config.floatX
np_fX = np.dtype(fX)


def as_fX(x):
    """
    convert input to value with type floatX
    """
    if isinstance(x, (float, int, long)):
        return np.array(x, dtype=fX)
    elif isinstance(x, np.ndarray):
        if x.dtype == np_fX:
            # don't make a copy if not necessary
            return x
        else:
            return x.astype(fX)
    else:
        # assume theano variable
        return x.astype(fX)


def is_nonshared_variable(x):
    return isinstance(x, theano.gof.graph.Variable)


def is_shared_variable(x):
    return isinstance(x, theano.compile.sharedvalue.SharedVariable)


def is_variable(x):
    return is_nonshared_variable(x) or is_shared_variable(x)


def all_equal(seq):
    """
    whether or not all elements of a sequence are equal
    """
    return len(set(seq)) == 1


def identity(x):
    return x


def first(f, *args):
    return f


def rectify(x, negative_coefficient=0):
    """
    general fast way of performing ReLU-type activations

    see below for info:
    https://github.com/Lasagne/Lasagne/pull/163
    https://github.com/Theano/Theano/issues/2698
    """
    f1 = 0.5 * (1 + negative_coefficient)
    f2 = 0.5 * (1 - negative_coefficient)
    return f1 * x + f2 * abs(x)


def stable_softmax(x):
    """
    numerical stabilization to avoid f32 overflow
    http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
    """
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out


def squared_error(pred, target):
    return (pred - target) ** 2


def categorical_crossentropy_i32(pred, target):
    """
    like theano.tensor.nnet.categorical_crossentropy, but with clearer
    assertions on the expected input
    """
    assert target.dtype == "int32"
    assert target.ndim == 1
    assert pred.dtype == fX
    assert pred.ndim == 2
    return T.nnet.categorical_crossentropy(pred, target)


def deep_clone(output, replace, **kwargs):
    """
    like theano.clone, but makes sure to replace in the default_update of
    shared variables as well
    """
    new_output = list(output)
    default_update_idxs = []
    for idx, v in enumerate(theano.gof.graph.inputs(output)):
        if hasattr(v, "default_update"):
            new_output.append(v.default_update)
            default_update_idxs.append(idx)
    cloned = theano.clone(new_output, replace, **kwargs)
    cloned_output = cloned[:len(output)]
    cloned_default_updates = cloned[len(output):]
    assert len(cloned_default_updates) == len(default_update_idxs)
    cloned_inputs = theano.gof.graph.inputs(cloned_output)
    for idx, update in zip(default_update_idxs, cloned_default_updates):
        v = cloned_inputs[idx]
        assert hasattr(v, "default_update")
        v.default_update = update
    return cloned_output


def shared_empty(ndim, dtype, name=None):
    """
    create shared variable with placeholder data
    """
    return theano.shared(np.zeros([1] * ndim, dtype=dtype), name=name)


def local_computation_output_length(input_size,
                                    local_size,
                                    stride,
                                    pad):
    """
    calculates the output size along a single axis for a convolutional or
    pooling operation
    """
    if input_size is None:
        return None

    without_stride = input_size + 2 * pad - local_size + 1
    # equivalent to np.ceil(without_stride / stride)
    output_size = (without_stride + stride - 1) // stride
    return output_size


def local_computation_output_shape(input_shape,
                                   axes,
                                   local_sizes,
                                   strides,
                                   pads):
    output_shape = list(input_shape)
    for axis, local_size, stride, pad in zip(axes,
                                             local_sizes,
                                             strides,
                                             pads):
        output_shape[axis] = local_computation_output_length(input_shape[axis],
                                                             local_size,
                                                             stride,
                                                             pad)
    return tuple(output_shape)

# ##################### utils for dealing with networks #####################


def nth_non_batch_axis(network, n):
    """
    returns the n-th axis that isn't the batch axis

    use cases: knowing which axis is the "feature" axis
    """
    batch_axis = network.find_hyperparameter(["batch_axis"])
    if batch_axis >= n:
        return n + 1
    else:
        return n


def find_axes(network,
              ndim,
              positive_keys,
              negative_keys,
              positive_default=None,
              negative_default=None):
    """
    given hyperparameters for positive axes (axes which to include) and
    negative axes (axes which to exclude), finds the axes < ndim that
    match
    """
    # TODO maybe rename function?
    pos = network.find_hyperparameter(positive_keys, None)
    neg = network.find_hyperparameter(negative_keys, None)
    # at most one should be set
    assert (pos is None) or (neg is None)

    if (pos is None) and (neg is None):
        # need defaults
        # exactly one should be set
        assert (positive_default is None) != (negative_default is None)
        if positive_default is not None:
            pos = positive_default
        else:
            neg = negative_default

    if pos is not None:
        return tuple(pos)
    else:
        return tuple([idx for idx in range(ndim) if idx not in neg])


def monitor(network, default=True):
    return network.find_hyperparameter(["monitor"], default)
