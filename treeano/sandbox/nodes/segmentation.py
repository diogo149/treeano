import theano
import theano.tensor as T


def approximate_negative_iou(pred, target, epsilon=1e-3):
    """
    differentiable approximation to negative IOU
    """
    intersection = (pred * target).sum(axis=(1, 2, 3))
    intersection = T.maximum(intersection, epsilon)
    union = T.maximum(pred, target).sum(axis=(1, 2, 3))
    union = T.maximum(union, epsilon)
    return -1.0 * (intersection / union).mean()
