import functools
import theano
import theano.tensor as T
import treeano


def approximate_negative_iou(pred, target, epsilon=1e-3):
    """
    differentiable approximation to negative IOU
    """
    intersection = (pred * target).sum(axis=(1, 2, 3))
    intersection = T.maximum(intersection, epsilon)
    union = T.maximum(pred, target).sum(axis=(1, 2, 3))
    union = T.maximum(union, epsilon)
    return -1.0 * (intersection / union).mean()


def hard_bootstrapping_binary_crossentropy(pred,
                                           target,
                                           num_taken,
                                           num_skipped=0,
                                           weight=1):
    """
    from
    High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks
    http://arxiv.org/abs/1604.04339
    """
    pixel_loss = treeano.utils.weighted_binary_crossentropy(pred,
                                                            target,
                                                            weight=weight)
    flat_loss = pixel_loss.flatten(2)
    sorted_flat_loss = T.sort(flat_loss)
    chosen_loss = sorted_flat_loss[:, -(num_taken + num_skipped):-num_skipped]
    return chosen_loss


def hard_bootstrapping_binary_crossentropy_fn(num_taken,
                                              num_skipped=0,
                                              weight=1):
    return functools.partial(hard_bootstrapping_binary_crossentropy,
                             num_taken=num_taken,
                             num_skipped=num_skipped,
                             weight=weight)


def hard_bootstrap_aggregator(loss, num_taken, num_skipped=0):
    flat_loss = loss.flatten(2)
    sorted_flat_loss = T.sort(flat_loss, axis=1)
    chosen_loss = sorted_flat_loss[:, -(num_taken + num_skipped):-num_skipped]
    return chosen_loss.mean()


def hard_bootstrap_aggregator_fn(num_taken, num_skipped=0):
    return functools.partial(hard_bootstrap_aggregator,
                             num_taken=num_taken,
                             num_skipped=num_skipped)


def mixed_hard_bootstrap_aggregator(loss, mix_rate, num_taken, num_skipped=0):
    flat_loss = loss.flatten(2)
    sorted_flat_loss = T.sort(flat_loss, axis=1)
    chosen_loss = sorted_flat_loss[:, -(num_taken + num_skipped):-num_skipped]
    return mix_rate * chosen_loss.mean() + (1 - mix_rate) * loss.mean()


def mixed_hard_bootstrap_aggregator_fn(mix_rate, num_taken, num_skipped=0):
    return functools.partial(mixed_hard_bootstrap_aggregator,
                             mix_rate=mix_rate,
                             num_taken=num_taken,
                             num_skipped=num_skipped)
