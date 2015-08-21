from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import canopy


def test_piecewise_linear_schedule():
    s = canopy.schedules.PiecewiseLinearSchedule([(2, 10),
                                                  (4, 15),
                                                  (7, -2)])
    ans = np.array([10,
                    10,
                    12.5,
                    15,
                    (2 * 15 + -2) / 3,
                    (15 + 2 * -2) / 3,
                    -2,
                    -2,
                    -2,
                    -2])
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)


def test_discrete_schedule():
    s = canopy.schedules.DiscreteSchedule([(2, 10),
                                           (4, 15),
                                           -2])
    ans = np.array([10, 10, 15, 15, -2, -2, -2, -2, -2, -2])
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)
