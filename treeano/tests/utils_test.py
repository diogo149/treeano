import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def test_maximum():
    assert treeano.utils.is_variable(treeano.utils.maximum(T.scalar(), 3))
    assert treeano.utils.is_variable(treeano.utils.maximum(3, T.scalar()))
    assert not treeano.utils.is_variable(treeano.utils.maximum(2, 3))
    assert not treeano.utils.is_variable(treeano.utils.maximum(2, np.ones(3)))


def test_stable_softmax():
    x = theano.shared(np.random.randn(50, 50).astype(fX))
    s1 = T.nnet.softmax(x).eval()
    s2 = treeano.utils.stable_softmax(x).eval()
    np.testing.assert_equal(s1, s2)


def test_stable_softmax_grad():
    x = theano.shared(np.random.randn(50, 50).astype(fX))
    s1 = T.nnet.softmax(x)
    s2 = treeano.utils.stable_softmax(
        x.reshape([50, 1, 5, 10]),
        axis=(2, 3)
    ).reshape([50, 50])
    np.testing.assert_allclose(s1.eval(), s2.eval(), rtol=1e-5)
    g1 = T.grad(s1[:10, :10].sum(), x)
    g2 = T.grad(s2[:10, :10].sum(), x)
    np.testing.assert_allclose(g1.eval(), g2.eval(), rtol=1e-5)


def test_linspace():
    np.testing.assert_allclose(np.linspace(-2, 42, 10).astype(fX),
                               treeano.utils.linspace(-2, 42, 10).eval(),
                               rtol=1e-5)


def _clone_test_case(clone_fn):
    x = T.matrix("x")
    y = T.matrix("y")
    x_shape = x.shape
    sample_y = np.random.randn(4, 5).astype(theano.config.floatX)
    srng = theano.tensor.shared_randomstreams.RandomStreams()
    mask = srng.binomial(n=1, p=0.5, size=x_shape)
    mask2 = clone_fn([mask], replace={x: y})[0]
    mask2.eval({y: sample_y})  # ERROR


@nt.raises(Exception)
def test_clone():
    """
    NOTE: if this test eventually passes (eg. theano fixes the issue),
    deep_clone may no longer be necessary
    """
    _clone_test_case(theano.clone)


def test_deep_clone():
    _clone_test_case(treeano.utils.deep_clone)


def test_find_axes():
    def axes(ndim, pos, neg):
        network = tn.HyperparameterNode(
            "a",
            tn.InputNode("b", shape=()),
            pos=pos,
            neg=neg,
        ).network()["a"]
        return treeano.utils.find_axes(network, ndim, ["pos"], ["neg"])

    @nt.raises(AssertionError)
    def axes_raises(*args):
        axes(*args)

    nt.assert_equal(axes(3, [2], None), (2,))
    nt.assert_equal(axes(3, [1], None), (1,))
    nt.assert_equal(axes(3, None, [1]), (0, 2))
    nt.assert_equal(axes(3, None, [0, 1]), (2,))
    axes_raises(3, [2], [1])


def test_binary_hinge_loss():
    x = np.array([[-1.5, -1, -0.5, 0, 0.5, 1, 1.5]] * 2, dtype=fX)
    y = np.array([[0] * 7, [1] * 7], dtype=fX)
    res = treeano.utils.binary_hinge_loss(T.constant(x),
                                          T.constant(y)).eval()
    ans = np.array([[0, 0, 0.5, 1, 1.5, 2, 2.5],
                    [2.5, 2, 1.5, 1, 0.5, 0, 0]],
                   dtype=fX)
    np.testing.assert_equal(res, ans)


def test_multiclass_hinge_loss():
    x = np.array([[0, 1],
                  [1, 0],
                  [0.5, 1.5],
                  [0, 0.5]] * 2,
                 dtype=fX)
    y = np.array([0] * 4 + [1] * 4, dtype="int32")
    res = treeano.utils.multiclass_hinge_loss(T.constant(x),
                                              T.constant(y)).eval()
    ans = np.array([[1, 2],
                    [1, 0],
                    [1, 2],
                    [1, 1.5],
                    [0, 1],
                    [2, 1],
                    [0, 1],
                    [0.5, 1]],
                   dtype=fX)
    np.testing.assert_equal(res, ans)


def test_root_mean_square():
    x = np.array([[3, 4],
                  [6, 8],
                  [1, 1]], dtype=fX)
    res = treeano.utils.root_mean_square(x, axis=1).eval()
    ans = np.array([5 / np.sqrt(2), 10 / np.sqrt(2), 1], dtype=fX)
    np.testing.assert_allclose(ans, res)


def test_is_float_ndarray():
    assert not treeano.utils.is_float_ndarray(3)
    assert not treeano.utils.is_float_ndarray([True])
    assert not treeano.utils.is_float_ndarray([42])
    assert not treeano.utils.is_float_ndarray(42.0)
    assert treeano.utils.is_float_ndarray(theano.shared(42.0).get_value())
    # NOTE: not including dscalar, because warn_float64 might be set
    for x in [T.scalar(), T.fscalar()]:
        assert treeano.utils.is_float_ndarray(x.eval({x: 42}))
    assert treeano.utils.is_float_ndarray(np.random.randn(42))
    assert treeano.utils.is_float_ndarray(
        np.random.randn(42).astype(np.float32))
    assert treeano.utils.is_float_ndarray(
        np.random.randn(42).astype(np.float64))


def test_is_int_ndarray():
    assert not treeano.utils.is_int_ndarray(3)
    assert not treeano.utils.is_int_ndarray([True])
    assert not treeano.utils.is_int_ndarray([42])
    assert not treeano.utils.is_int_ndarray(42.0)
    for x in [T.iscalar(), T.lscalar()]:
        assert treeano.utils.is_int_ndarray(x.eval({x: 42}))
    assert treeano.utils.is_int_ndarray(np.random.randint(42, size=(4, 5)))
    assert treeano.utils.is_int_ndarray(
        np.random.randint(42, size=(4, 5)).astype(np.int32))
