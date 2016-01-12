import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

from treeano.theano_extensions.irregular_length import ungroup_irregular_length_tensors

fX = theano.config.floatX


def test_ungroup_irregular_length_tensors():
    x = np.array([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9]],
                 dtype=fX)
    lengths = np.array([2, 1, 0, 2])
    ans = np.array([[[0, 1],
                     [2, 3]],
                    [[4, 5],
                     [0, 0]],
                    [[0, 0],
                     [0, 0]],
                    [[6, 7],
                     [8, 9]]])
    res = ungroup_irregular_length_tensors(x, lengths).eval()
    np.testing.assert_equal(ans, res)


def test_ungroup_irregular_length_tensors_grad():
    x = np.array([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9]],
                 dtype=fX)
    x = T.constant(x)
    lengths = np.array([2, 1, 0, 2])
    ans = np.array([[1, 1],
                    [1, 1],
                    [2, 2],
                    [0, 0],
                    [0, 0]],
                   dtype=fX)
    ungrouped = ungroup_irregular_length_tensors(x, lengths)
    out = ungrouped[0].sum() + 2 * ungrouped[1].sum()
    grad = T.grad(out, x).eval()
    np.testing.assert_equal(ans, grad)


def test_ungroup_irregular_length_tensors_numeric_gradient():
    lengths = np.array([2, 3, 4, 5, 7, 2], dtype=fX)
    T.verify_grad(lambda x: ungroup_irregular_length_tensors(x, lengths),
                  [np.random.randn(23, 10).astype(fX)],
                  rng=np.random)
