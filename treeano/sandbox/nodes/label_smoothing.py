import functools
import theano
import theano.tensor as T

fX = theano.config.floatX


def label_smoothing_categorical_crossentropy(pred,
                                             target,
                                             alpha,
                                             beta=None,
                                             num_classes=None):
    if target.dtype == "int32":
        assert pred.ndim - 1 == target.ndim
        assert target.ndim == 1
        assert pred.dtype == fX
        assert pred.ndim == 2
        target = T.extra_ops.to_one_hot(target, nb_class=num_classes, dtype=fX)
    if beta is None:
        beta = (1.0 - alpha) / (num_classes - 1)
    return T.nnet.categorical_crossentropy(pred,
                                           T.clip(target, beta, alpha))


def label_smoothing_categorical_crossentropy_fn(alpha,
                                                beta=None,
                                                num_classes=None):
    return functools.partial(label_smoothing_categorical_crossentropy,
                             alpha=alpha,
                             beta=beta,
                             num_classes=num_classes)
