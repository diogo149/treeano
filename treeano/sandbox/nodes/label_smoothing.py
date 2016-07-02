import functools
import theano
import theano.tensor as T

fX = theano.config.floatX


def label_smoothing_categorical_crossentropy(pred, target, alpha, beta):
    if target.dtype == "int32":
        assert pred.ndim - 1 == target.ndim
        assert target.ndim == 1
        assert pred.dtype == fX
        assert pred.ndim == 2
        target = T.extra_ops.to_one_hot(target, nb_class=target, dtype=fX)
    return T.nnet.categorical_crossentropy(pred,
                                           T.clip(target, beta, alpha))


def label_smoothing_categorical_crossentropy_fn(alpha, beta):
    return functools.partial(label_smoothing_categorical_crossentropy,
                             alpha=alpha,
                             beta=beta)
