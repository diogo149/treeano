import numpy as np
import scipy.sparse
import theano
import theano.tensor as T
import treeano


def coocurrence_matrix(data, vocabulary_size, window_size=5):
    """
    default hyperparameters based on insights from
    "Improving Distributional Similarity with Lessons Learned from Word Embeddings"
    - no dynamic context window
    - no subsampling
    - no deleting rare words

    idxs:
    list of list of indexes
    """
    m = scipy.sparse.lil_matrix((vocabulary_size,) * 2)
    windows = list(range(-window_size, window_size + 1))
    windows.remove(0)
    for sentence in data:
        for i, idx1 in enumerate(sentence):
            for w in windows:
                j = i + w
                if 0 <= j < len(sentence):
                    idx2 = sentence[j]
                    m[idx1, idx2] += 1
    return m


def pointwise_mutual_information(counts_both,
                                 row_counts=None,
                                 col_counts=None,
                                 count_total=None):
    """
    pmi calculation with optional row, column, and total counts
    """
    pmi = 0
    if count_total is not None:
        pmi = T.log(count_total)
    if row_counts is not None:
        pmi = pmi - T.log(row_counts.dimshuffle(0, "x"))
    if col_counts is not None:
        pmi = pmi - T.log(col_counts.dimshuffle("x", 0))
    pmi = pmi + T.log(counts_both)
    return pmi


def skipgram_cost(preds,
                  counts_both,
                  row_counts,
                  col_counts,
                  alpha=0.75,
                  epsilon=1e-8):
    """
    skip-gram cost with softmax
    based on "Distributed Representations Of Words And Phrases And Their Compositionality"
    some implementation from "Improving Distributional Similarity with Lessons Learned from Word Embeddings"

    preds:
    prediction matrix of size m x n

    counts_both:
    pairwise count matrix of size m x n

    row_counts:
    counts of words corresponding to rows of preds (length m)

    col_counts:
    counts of words corresponding to columns of preds (length n)

    NOTE: assumes that the number of columns is the vocabulary size
    """

    nonzero = counts_both > epsilon
    # words are kept with probability proportional to 1/sqrt(frequency)
    freq_prod = row_counts.dimshuffle(0, "x") * col_counts.dimshuffle("x", 0)
    weight = 1 / T.sqrt(freq_prod)
    weight = nonzero * weight
    # for noise distribution, unigram distribution raised to 3/4 is used
    # also called "context distribution smoothing"

    pred_probs = treeano.utils.stable_softmax(preds)
    true_probs = counts_both / counts_both.sum(axis=1, keepdims=True)
    cross_entropy = -true_probs * T.log(pred_probs)
    # mean over minibatch
    cost = (weight * cross_entropy).sum(axis=1).mean()
    return cost


def sgns_cost(preds,
              counts_both,
              row_counts=None,
              col_counts=None,
              count_total=None,
              epsilon=1e-8):
    """
    skip-gram negative sampling cost
    based on "Distributed Representations Of Words And Phrases And Their Compositionality"
    formula from "Swivel: Improving Embeddings by Noticing What's Missing"

    preds:
    prediction matrix of size m x n

    counts_both:
    pairwise count matrix of size m x n

    row_counts:
    counts of words corresponding to rows of preds (length m)

    col_counts:
    counts of words corresponding to columns of preds (length n)

    count_total:
    scalar corresponding to the count of all words in the dataset
    """
    nonzero = counts_both > epsilon
    # TODO add weighting - swivel paper doesn't specify which to use
    weight = nonzero
    # add epsilon to stabilize log for 0 counts
    target = pointwise_mutual_information(
        counts_both=counts_both + epsilon,
        row_counts=row_counts,
        col_counts=col_counts,
        count_total=count_total)
    # mean over minibatch
    loss = T.sum(weight * T.sqr(preds - target), axis=1).mean()
    return loss


def glove_cost(preds,
               counts_both,
               x_max=100,
               alpha=0.75,
               epsilon=1e-8):
    """
    based on "GloVe: Global Vectors For Word Representation"
    formula from "Swivel: Improving Embeddings by Noticing What's Missing"

    preds:
    prediction matrix of size m x n

    counts_both:
    pairwise count matrix of size m x n
    """
    # give 0 weight where count == 0
    nonzero = counts_both > epsilon
    weight = T.maximum(T.pow(counts_both / x_max, alpha), 1) * nonzero
    # add epsilon to stabilize log for 0 counts
    target = T.log(counts_both + epsilon)
    # mean over minibatch
    loss = T.sum(weight * T.sqr(preds - target), axis=1).mean()
    return loss


def swivel_cost(preds,
                counts_both,
                row_counts=None,
                col_counts=None,
                count_total=None,
                epsilon=1e-8):
    """
    based on "Swivel: Improving Embeddings by Noticing What's Missing"

    preds:
    prediction matrix of size m x n

    counts_both:
    pairwise count matrix of size m x n

    row_counts:
    counts of words corresponding to rows of preds (length m)

    col_counts:
    counts of words corresponding to columns of preds (length n)

    count_total:
    scalar corresponding to the count of all words in the dataset
    """
    # smoothed point-wise mutual information
    smoothed_pmi = pointwise_mutual_information(
        counts_both=T.maximum(counts_both, 1),
        row_counts=row_counts,
        col_counts=col_counts,
        count_total=count_total)

    # monotonically increasing confidence function
    confidence = T.sqrt(counts_both)

    squared_error = 0.5 * confidence * T.sqr(preds - smoothed_pmi)

    soft_hinge = T.log(1 + T.exp(preds - smoothed_pmi))

    gt0 = counts_both > epsilon

    # combine piece-wise loss
    loss = gt0 * squared_error + (1 - gt0) * soft_hinge
    # mean over minibatch
    loss = T.sum(loss, axis=1).mean()
    return loss
