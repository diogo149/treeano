"""
from: https://gist.github.com/eickenberg/f1a0e368961ef6d05b5b
by Michael Eickenberg
"""

import numpy as np
from treeano.theano_extensions import mgrid, ogrid


def test_mgrid_ogrid():
    fmgrid = np.mgrid[0:1:.1, 1:10:1., 10:100:10.]
    imgrid = np.mgrid[0:2:1, 1:10:1, 10:100:10]

    fogrid = np.ogrid[0:1:.1, 1:10:1., 10:100:10.]
    iogrid = np.ogrid[0:2:1, 1:10:1, 10:100:10]

    tfmgrid = mgrid[0:1:.1, 1:10:1., 10:100:10.]
    timgrid = mgrid[0:2:1, 1:10:1, 10:100:10]

    tfogrid = ogrid[0:1:.1, 1:10:1., 10:100:10.]
    tiogrid = ogrid[0:2:1, 1:10:1, 10:100:10]

    for g1, g2 in zip([fmgrid, imgrid, fogrid, iogrid],
                      [tfmgrid, timgrid, tfogrid, tiogrid]):
        for v1, v2 in zip(g1, g2):
            np.testing.assert_almost_equal(v1, v2.eval(), decimal=6)
