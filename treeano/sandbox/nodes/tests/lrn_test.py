import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import lrn

fX = theano.config.floatX


def ground_truth_normalizer(bc01, k, n, alpha, beta):
    """
    This code is adapted from pylearn2.
    https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def ground_truth_normalize_row(row, k, n, alpha, beta):
        assert row.ndim == 1
        out = np.zeros(row.shape)
        for i in range(row.shape[0]):
            s = k
            tot = 0
            for j in range(max(0, i - n // 2),
                           min(row.shape[0], i + n // 2 + 1)):
                tot += 1
                sq = row[j] ** 2.
                assert sq > 0.
                assert s >= k
                assert alpha > 0.
                s += alpha * sq
                assert s >= k
            assert tot <= n
            assert s >= k
            s = s ** beta
            out[i] = row[i] / s
        return out

    c01b = bc01.transpose(1, 2, 3, 0)
    out = np.zeros(c01b.shape)

    for r in range(out.shape[1]):
        for c in range(out.shape[2]):
            for x in range(out.shape[3]):
                out[:, r, c, x] = ground_truth_normalize_row(
                    row=c01b[:, r, c, x],
                    k=k, n=n, alpha=alpha, beta=beta)
    out_bc01 = out.transpose(3, 0, 1, 2)
    return out_bc01


def test_local_response_normalization_2d():
    vw = treeano.VariableWrapper("foo",
                                 variable=T.tensor4(),
                                 shape=(3, 4, 5, 6))
    kwargs = dict(
        # use a big value of alpha so mistakes involving alpha show up strong
        alpha=1.5,
        k=2,
        beta=0.75,
        n=5,
    )
    fn = theano.function([vw.variable],
                         [lrn.local_response_normalization_2d(vw, **kwargs)])
    x = np.random.randn(3, 4, 5, 6).astype(fX)
    res, = fn(x)
    ans = ground_truth_normalizer(x, **kwargs)
    np.testing.assert_allclose(ans, res, rtol=1e-5)
