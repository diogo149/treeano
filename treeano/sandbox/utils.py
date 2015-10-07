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
