import numpy as np
import theano
import theano.tensor as T

from treeano.sandbox.nodes import triplet_network as trip

fX = theano.config.floatX


def test_triplet_network_indices():
    for y in [np.random.randint(0, 20, 300).astype(np.int32),
              np.random.randint(0, 2, 256).astype(np.int32),
              np.array([0, 1, 0, 1, 1, 1, 0], dtype=np.int32),
              np.array([0, 0, 1, 1], dtype=np.int32)]:
        y_true = T.constant(y)
        symbolic_idxs = trip.symmetric_idxs(y_true)
        idxs = symbolic_idxs.eval()
        assert np.alltrue(y[idxs[:, 0]] == y[idxs[:, 1]])
        assert np.alltrue(y[idxs[:, 2]] != y[idxs[:, 1]])


def test_classification_triplet_loss():
    # NOTE: can be slow if compilation not cached
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    y_true = T.constant(y)
    embeddings = theano.shared(np.random.randn(4, 128).astype(fX))
    loss = trip.deep_metric_learning_classification_triplet_loss(
        embeddings, trip.symmetric_idxs(y_true))
    grad = T.grad(loss, [embeddings])[0]
    # SGD
    new_embeddings = (embeddings - 0.01 * grad)
    # set embeddings to have norm of 1
    new_embeddings2 = (new_embeddings
                       / trip.l2_norm(new_embeddings, axis=1, keepdims=True))
    fn = theano.function([], [loss], updates={embeddings: new_embeddings2})
    prev_loss = np.inf
    for _ in range(200):
        l = fn()[0]
        assert l < prev_loss
    import scipy.spatial.distance
    vecs = embeddings.get_value()
    dist_matrix = scipy.spatial.distance.cdist(vecs, vecs)

    for row, same, differents in [(0, 1, (2, 3)),
                                  (1, 0, (2, 3)),
                                  (2, 3, (0, 1)),
                                  (3, 2, (0, 1))]:
        for different in differents:
            assert dist_matrix[row, same] < dist_matrix[row, different], dict(
                dist_matrix=dist_matrix,
                row=row,
                same=same,
                different=different,
            )


def test_classification_triplet_same_label():
    # NOTE: can be slow if compilation not cached
    # test what happens when there are no triplets (all have the same label)
    y = np.array([0, 0, 0], dtype=np.int32)
    y_true = T.constant(y)
    embeddings = theano.shared(np.random.randn(3, 128).astype(fX))
    loss = trip.deep_metric_learning_classification_triplet_loss(
        embeddings, trip.symmetric_idxs(y_true))
    loss.eval()


def test_classification_triplet_same_embedding():
    # NOTE: can be slow if compilation not cached
    # test what happens when all triplets have the same label
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    y_true = T.constant(y)
    embeddings = theano.shared(np.zeros((4, 128), dtype=fX))
    loss = trip.deep_metric_learning_classification_triplet_loss(
        embeddings, trip.symmetric_idxs(y_true))
    embeddings_g = T.grad(loss, embeddings)
    embeddings_g.eval()
