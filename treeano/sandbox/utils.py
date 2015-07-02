import theano
import theano.tensor as T


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
