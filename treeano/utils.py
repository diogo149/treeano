import numbers
import functools

import numpy as np
import theano
import theano.tensor as T

fX = theano.config.floatX
np_fX = np.dtype(fX)


def as_fX(x):
    """
    convert input to value with type floatX
    """
    if isinstance(x, (float, numbers.Integral)):
        return np.array(x, dtype=fX)
    elif isinstance(x, np.ndarray):
        if x.dtype == np_fX:
            # don't make a copy if not necessary
            return x
        else:
            return x.astype(fX)
    else:
        # assume theano variable
        if x.dtype == fX:
            return x
        else:
            return x.astype(fX)


def is_nonshared_variable(x):
    return isinstance(x, theano.gof.graph.Variable)


def is_shared_variable(x):
    return isinstance(x, theano.compile.sharedvalue.SharedVariable)


def is_variable(x):
    return is_nonshared_variable(x) or is_shared_variable(x)


def is_number(x):
    return isinstance(x, numbers.Number)


def is_integral(x):
    return isinstance(x, numbers.Integral)


def is_ndarray(x):
    return isinstance(x, np.ndarray)


def is_float_ndarray(x):
    return is_ndarray(x) and issubclass(x.dtype.type, np.floating)


def is_int_ndarray(x):
    return is_ndarray(x) and issubclass(x.dtype.type, numbers.Integral)


def all_equal(seq):
    """
    whether or not all elements of a sequence are equal
    """
    return len(set(seq)) == 1


def identity(x):
    return x


def first(f, *args):
    return f


def maximum(a, b):
    """
    polymorphic version of max or theano.tensor.maximum
    """
    if is_variable(a) or is_variable(b):
        return T.maximum(a, b)
    else:
        return np.maximum(a, b)


def sign_non_zero(x):
    """
    returns the sign of the input variable as 1 or -1

    arbitrarily sign_non_zero(0) = 1
    """
    return 2 * (x >= 0) - 1


def rectify(x, negative_coefficient=0):
    """
    general way of performing ReLU-type activations
    """
    return T.nnet.relu(x, alpha=negative_coefficient)


def newaxis(x, axis):
    """
    inserts a new broadcastable axis into a tensor
    """
    axes = list(range(x.ndim))
    axes.insert(axis, "x")
    return x.dimshuffle(*axes)


def root_mean_square(x, axis=None):
    return T.sqrt(T.mean(T.sqr(x), axis=axis))


def stable_softmax(x, axis=1):
    """
    numerical stabilization to avoid f32 overflow
    http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
    """
    # TODO test performance on axis
    # if this way is slow, could reshape, do the softmax, then reshape back
    if axis == tuple(range(1, x.ndim)):
        # reshape, do softmax, then reshape back, in order to be differentiable
        # TODO could do reshape trick for any set of sequential axes
        # that end with last (eg. 2,3), not only when starting with axis 1
        return stable_softmax(x.flatten(2)).reshape(x.shape)
    else:
        e_x = T.exp(x - x.max(axis=axis, keepdims=True))
        out = e_x / e_x.sum(axis=axis, keepdims=True)
        return out


def squared_error(pred, target):
    return (pred - target) ** 2


def absolute_error(pred, target):
    return abs(pred - target)


def binary_hinge_loss(pred, target):
    """
    assumes that t is in {0, 1}
    """
    # convert to -1/1
    target = 2 * target - 1
    return rectify(1 - target * pred)


def binary_squared_hinge_loss(pred, target):
    return T.sqr(binary_hinge_loss(pred, target))


def multiclass_hinge_loss(pred, target):
    """
    Weston Watkins formulation

    assumes that pred has shape (something, number of classes)
    """
    assert target.dtype == "int32"
    assert target.ndim == 1
    assert pred.dtype == fX
    assert pred.ndim == 2
    # NOTE: this uses AdvancedSubtensor, which may be slow!
    target_pred = pred[T.arange(pred.shape[0]), target].dimshuffle(0, "x")
    return rectify(pred - target_pred + 1)


def multiclass_squared_hinge_loss(pred, target):
    return T.sqr(multiclass_hinge_loss(pred, target))


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


def weighted_binary_crossentropy(output, target, weight=1):
    """
    binary cross-entropy with a different weight for the positive class
    """
    if weight == 1:
        return T.nnet.binary_crossentropy(output, target)
    else:
        return -(weight * target * T.log(output)
                 + (1.0 - target) * T.log(1.0 - output))


def weighted_binary_crossentropy_fn(weight):
    return functools.partial(weighted_binary_crossentropy, weight=weight)


def linspace(start, stop, num):
    """
    like numpy.linspace
    """
    range_01 = as_fX(T.arange(num)) / as_fX(num - 1)
    return start + range_01 * (stop - start)


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


def seed_MRG_RandomStreams(rng, seed=12345):
    """
    seeds random state of theano.sandbox.rng_mrg.MRG_RandomStreams
    """
    rng.rstate = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=seed).rstate

# ############################## smart reducing ##############################


def smart_reduce(op, iterable):
    iterable = list(iterable)
    if len(iterable) == 1:
        return iterable[0]
    else:
        return functools.reduce(op, iterable[1:], iterable[0])


def smart_add(x, y):
    """
    0-aware add, to prevent computation graph from getting very large
    """
    if x == 0:
        return y
    elif y == 0:
        return x
    else:
        return x + y


def smart_mul(x, y):
    """
    0- and 1- aware multiply, to prevent computation graph from getting very
    large
    """
    if x == 0 or y == 0:
        return 0
    elif x == 1:
        return y
    elif y == 1:
        return x
    else:
        return x * y

smart_sum = functools.partial(smart_reduce, smart_add)
smart_product = functools.partial(smart_reduce, smart_mul)

# ##################### utils for dealing with variable wrappers ############


def vw_reduce_shape(vws):
    """
    checks that all vws have the same shape (or are broadcastable with each
    other) and returns what the shape would be
    """
    assert all_equal([vw.ndim for vw in vws])
    shape = []
    for i in range(vws[0].ndim):
        sizes = [vw.shape[i]
                 for vw in vws
                 if not vw.broadcastable[i]]
        assert all_equal(sizes)
        if sizes:
            shape.append(sizes[0])
        else:
            shape.append(1)
    return tuple(shape)

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
