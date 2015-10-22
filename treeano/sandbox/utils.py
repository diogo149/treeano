import abc

import six
import theano
import theano.tensor as T

fX = theano.config.floatX


def cross_covariance(y, z):
    """
    from "Discovering Hidden Factors of Variation in Deep Networks"
    http://arxiv.org/abs/1412.6583
    """
    y_mean = T.mean(y, axis=0, keepdims=True)
    z_mean = T.mean(z, axis=0, keepdims=True)
    y_centered = y - y_mean  # (n, i)
    z_centered = z - z_mean  # (n, j)
    outer_prod = (y_centered.dimshuffle(0, 1, 'x') *
                  z_centered.dimshuffle(0, 'x', 1))  # (n, i, j)
    C = 0.5 * T.sum(T.sqr(T.mean(outer_prod, axis=0)))
    return C


def soft_categorical_crossentropy_i32(pred, target, alpha=0.01):
    """
    softer cross-entropy function, where target is treated at not being
    exactly 1, and instead spreading that probability uniformly to other
    classes
    """
    assert target.dtype == "int32"
    assert target.ndim == 1
    assert pred.dtype == fX
    assert pred.ndim == 2
    nb_class = pred.shape[1]
    alpha = 0.01
    t = T.extra_ops.to_one_hot(target, nb_class=nb_class, dtype=pred.dtype)
    t = T.clip(t, alpha / (nb_class.astype(fX) - 1.0), 1 - alpha)
    return T.nnet.categorical_crossentropy(pred, t)


def soft_binary_crossentropy(pred, target, alpha=0.01):
    """
    softer binary cross-entropy function, where target is treated at not being
    exactly 1
    """
    return T.nnet.binary_crossentropy(pred, T.clip(target, alpha, 1 - alpha))


class OverwriteGrad(six.with_metaclass(abc.ABCMeta, object)):

    """
    wraps a function, allowing a different gradient to be applied

    based on Lasagne Recipes on Guided Backpropagation
    """

    def __init__(self, fn):
        self.fn = fn
        # memoizes an OpFromGraph instance per tensor type
        self.ops = {}

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # note the tensor type of the input variable to the fn
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # create a suitable Op if not yet done
        if tensor_type not in self.ops:
            # create an input variable of the correct type
            inp = tensor_type()
            # pass it through the fn (and move to GPU if needed)
            outp = maybe_to_gpu(self.fn(inp))
            # fix the forward expression
            op = theano.OpFromGraph([inp], [outp])
            # replace the gradient with our own
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # apply the memoized Op to the input we got
        return self.ops[tensor_type](x)

    @abc.abstractmethod
    def grad(self, inputs, out_grads):
        pass
