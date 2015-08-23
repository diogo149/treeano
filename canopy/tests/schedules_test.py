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

    s = canopy.schedules.DiscreteSchedule([-2])
    ans = np.array([-2, -2, -2, -2, -2])
    res = np.array([s(None, None) for _ in range(5)])
    np.testing.assert_allclose(ans, res)


def test_step_schedule():
    s = canopy.schedules.StepSchedule(1, 2, [3, 5, 9])
    ans = np.array([1, 1, 2, 2, 4, 4, 4, 4, 8, 8])
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)


def test_recurring_step_schedule():
    s = canopy.schedules.RecurringStepSchedule(1, 2, 3)
    ans = np.array([1, 1, 2, 2, 2, 4, 4, 4, 8, 8])
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)


def test_inverse_decay_schedule():
    s = canopy.schedules.InverseDecaySchedule(1, 0.1, -2)
    ans = np.array([1, 1.1 ** 2, 1.2 ** 2, 1.3 ** 2, 1.4 ** 2])
    res = np.array([s(None, None) for _ in range(5)])
    np.testing.assert_allclose(ans, res)


def test_fixed_schedule():
    s = canopy.schedules.FixedSchedule(42)
    ans = np.array([42] * 10)
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)


def test_exponential_schedule():
    s = canopy.schedules.ExponentialSchedule(2.3, 0.7)
    ans = 2.3 * 0.7 ** np.arange(10)
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)


def test_half_life_schedule():
    s = canopy.schedules.HalfLifeSchedule(1, 2)
    ans = np.array([1, np.sqrt(0.5), 0.5, np.sqrt(0.125), 0.25])
    res = np.array([s(None, None) for _ in range(5)])
    np.testing.assert_allclose(ans, res)


def test_multi_stage_schedule():
    s = canopy.schedules.MultiStageSchedule(
        [(2, canopy.schedules.FixedSchedule(2)),
         (5, canopy.schedules.ExponentialSchedule(3, 2)),
         canopy.schedules.FixedSchedule(1)])
    ans = np.array([2, 2, 3, 6, 12, 1, 1, 1, 1, 1])
    res = np.array([s(None, None) for _ in range(10)])
    np.testing.assert_allclose(ans, res)
