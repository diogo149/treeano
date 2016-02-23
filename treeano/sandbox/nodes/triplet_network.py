from collections import defaultdict
import itertools

import numpy as np
import theano
import theano.tensor as T
from theano.compile.ops import as_op

# ################################## utils ##################################


def l2_norm(x, axis=None, keepdims=False):
    # Adding epsilon for numerical stability.
    # The gradient of sqrt at zero is otherwise undefined.
    epsilon = 1e-8
    return T.sqrt(T.sum(x ** 2, axis=axis, keepdims=keepdims) + epsilon)

# ################################# indices #################################


@as_op(itypes=[theano.tensor.ivector],
       otypes=[theano.tensor.imatrix])
def symmetric_idxs(targets):
    """
    uses a technique that generates all possible (same, same, different)
    triplets in a given minibatch

    rationale: using a normal triplet network, the entire network needs to run
    3 times to get 1 triplet for a total of batch_size / 3 triplets. instead,
    we look at every valid triplet in a minibatch, resulting in an expected
    O(batch_size^3) triplets, while also not requiring the user to change how
    minibatches are created (ie. no need to manually create the
    (same, same, different) pattern in each minibatch)

    NOTE: there may be a more efficient way to do this by precomputing the
    distance between each pair of embeddings in a minibatch
    (eg. O(b^2 * k + b^3)) instead of computing O(n^3) distances
    (eg. O(b^3 * k))
      b = batch size
      k = embedding size

    returns a matrix of the form
    [[pos_idx_1, ref_idx_1, neg_idx_1],
     ...
     [pos_idx_n, ref_idx_n, neg_idx_n]]
    """
    # group indices by class
    grouped = defaultdict(set)
    for idx, target in enumerate(targets):
        grouped[target].add(idx)
    # create triplets
    all_idxs = set(range(len(targets)))
    triples = []
    for pos_idxs in grouped.values():
        neg_idxs = all_idxs - pos_idxs
        for pos_idx, ref_idx in itertools.permutations(pos_idxs, 2):
            for neg_idx in neg_idxs:
                triples.append([pos_idx, ref_idx, neg_idx])
    if len(triples) > 0:
        return np.array(triples, dtype=np.int32)
    else:
        return np.zeros((0, 3), dtype=np.int32)

# ############################# loss calculation #############################


def deep_metric_learning_classification_triplet_loss(embeddings, idxs):
    """
    loss based on http://arxiv.org/abs/1412.6622

    embeddings:
    floatX matrix of shape (batch_size, embedding_size)

    idxs:
    n x 3 matrix of the form
    [[pos_idx_1, ref_idx_1, neg_idx_1],
     ...
     [pos_idx_n, ref_idx_n, neg_idx_n]]
    """
    positives = embeddings[idxs[:, 0]]
    references = embeddings[idxs[:, 1]]
    negatives = embeddings[idxs[:, 2]]
    pos_l2_norm = l2_norm(positives - references, axis=1)
    neg_l2_norm = l2_norm(negatives - references, axis=1)
    max_l2_norm = T.maximum(pos_l2_norm, neg_l2_norm)
    pos_exp_dist = T.exp(pos_l2_norm - max_l2_norm)
    neg_exp_dist = T.exp(neg_l2_norm - max_l2_norm)
    losses = pos_exp_dist / (pos_exp_dist + neg_exp_dist)
    loss_mean = T.mean(losses)
    return loss_mean


def facenet_triplet_loss(embeddings, idxs, alpha=0.2):
    """
    loss based on http://arxiv.org/abs/1503.03832

    alpha: margin
    """
    positives = embeddings[idxs[:, 0]]
    references = embeddings[idxs[:, 1]]
    negatives = embeddings[idxs[:, 2]]

    def norm_squared(x):
        return T.sum(T.sqr(x), axis=1)

    losses = T.nnet.relu(norm_squared(references - positives)
                         + alpha
                         - norm_squared(references - negatives))
    return T.mean(losses)
