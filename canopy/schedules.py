"""
schedules for canopy.handlers.schedule_hyperparameter
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


class FixedSchedule(object):

    def __init__(self, value):
        """
        returns a constant value
        """
        self.value = value

    def __call__(self, in_dict, previous_output_dict):
        return self.value


class PiecewiseLinearSchedule(object):

    def __init__(self, schedule):
        """
        takes in a schedule of the format list of tuples of iteration and
        hyperparameter value at that iteration
        eg.
        PiecewiseLinearSchedule([(1, 0.1),
                                 (1000, 0.01),
                                 (1200, 0.001)])

        NOTES:
        - count is 1-indexed
        - anything before the first pair is assumed to be constant
        - anything after the last pair is assumed to be constant
        """
        for p1, p2 in zip(schedule, schedule[1:]):
            # not sorting the schedule automatically because it would be
            # harder to read/understand in the client's code
            assert p1[0] < p2[0], "Sort your schedule!"
        self.schedule = schedule
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        self.num_ += 1
        if self.num_ <= self.schedule[0][0]:
            return self.schedule[0][1]
        elif self.num_ >= self.schedule[-1][0]:
            return self.schedule[-1][1]
        else:
            for (n1, v1), (n2, v2) in zip(self.schedule, self.schedule[1:]):
                if n1 <= self.num_ < n2:
                    return v1 + (v2 - v1) * (self.num_ - n1) / (n2 - n1)
            assert False


class DiscreteSchedule(object):

    def __init__(self, schedule):
        """
        takes in a schedule of the format list of tuples of iteration and
        hyperparameter value at that iteration (the iteration is the final
        iteration at that hyperparameter value), followed by the value after
        the last number of iterations
        eg.
        DiscreteSchedule([(3, 0.1),
                          (1000, 0.01),
                          0.001],)

        NOTES:
        - count is 1-indexed
        """
        for p1, p2 in zip(schedule, schedule[1:-1]):
            # not sorting the schedule automatically because it would be
            # harder to read/understand in the client's code
            assert p1[0] < p2[0], "Sort your schedule!"
        self.schedule = schedule
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        self.num_ += 1
        for n, v in self.schedule[:-1]:
            if self.num_ <= n:
                return v
        return self.schedule[-1]


class StepSchedule(object):

    def __init__(self, initial, step_factor, iters):
        """
        multiplies the current value by step_factor for each iteration in iters
        """
        self.initial = initial
        self.step_factor = step_factor
        self.iters = iters
        self.value_ = initial
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        self.num_ += 1
        if self.num_ in self.iters:
            self.value_ *= self.step_factor
        return self.value_


class RecurringStepSchedule(object):

    def __init__(self, initial, step_factor, frequency):
        """
        multiplies the current value by step_factor every frequency iterations
        """
        self.initial = initial
        self.step_factor = step_factor
        self.frequency = frequency
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        self.num_ += 1
        return (self.initial *
                self.step_factor ** np.floor(self.num_ / self.frequency))


class InverseDecaySchedule(object):

    def __init__(self, initial, gamma, power):
        """
        change value inversely proportionally to iteration
        """
        self.initial = initial
        self.gamma = gamma
        self.power = power
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        val = self.initial * (1 + self.gamma * self.num_) ** -self.power
        # add afterwards, so that initial value equals initial
        self.num_ += 1
        return val


class ExponentialSchedule(object):

    def __init__(self, initial, factor):
        """
        change value proportional to factor^iterations
        """
        self.initial = initial
        self.factor = factor
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        val = self.initial * self.factor ** self.num_
        # add afterwards, so that initial value equals initial
        self.num_ += 1
        return val


class HalfLifeSchedule(object):

    def __init__(self, initial, half_life):
        """
        halves the value that is output every half_life iterations

        the same as an ExponentialSchedule, with a potentially easier
        to reason about hyperparameter
        """
        self.initial = initial
        self.half_life = half_life
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        val = self.initial * 0.5 ** (self.num_ / self.half_life)
        # add afterwards, so that initial value equals initial
        self.num_ += 1
        return val


class MultiStageSchedule(object):

    def __init__(self, schedules):
        """
        takes in schedules of the format list of tuples of iteration and
        schedule , followed by the schedule after the last number of iterations

        similar to DiscreteSchedule

        eg.
        MultiStageSchedule([(3, FixedSchedule(2)),
                            (1000, ExponentialSchedule(3, 4)),
                            FixedSchedule(1)],)

        NOTES:
        - count is 1-indexed
        """
        for p1, p2 in zip(schedules, schedules[1:-1]):
            # not sorting the schedule automatically because it would be
            # harder to read/understand in the client's code
            assert p1[0] < p2[0], "Sort your schedule!"
        self.schedules = schedules
        self.num_ = 0

    def __call__(self, in_dict, previous_output_dict):
        self.num_ += 1
        for n, s in self.schedules[:-1]:
            if self.num_ <= n:
                return s(in_dict, previous_output_dict)
        return self.schedules[-1](in_dict, previous_output_dict)


class PiecewiseLogLinearSchedule(object):

    def __init__(self, schedule):
        """
        takes in a schedule of the format list of tuples of iteration and
        hyperparameter value at that iteration
        """
        self.schedule = schedule
        self.sub_schedule_ = PiecewiseLinearSchedule(
            [(n, np.log(v)) for n, v in schedule])

    def __call__(self, in_dict, previous_output_dict):
        res = self.sub_schedule_(in_dict, previous_output_dict)
        return np.exp(res)


class CyclicLinearSchedule(object):

    def __init__(self,
                 v0_initial,
                 v1_initial,
                 frequency,
                 boundary="mirror",
                 v0_decay=1,
                 v1_decay=1):
        """
        returns values cycling between v0 and v1

        v0_initial:
        initial value

        v1_initial:
        target value for initial cycle

        frequency:
        number of steps in between v0 and v1

        boundary:
        one of:
        - "wrap": when reaching v1, start at the new value of v0
        - "mirror": when reaching v1, linearly move towards v0

        v0_decay:
        value to multiply to v0 upon reaching v1

        v1_decay:
        value to multiply to v1 upon reaching v0
        """
        self.v0 = v0_initial
        self.v1 = v1_initial
        self.frequency = frequency
        self.boundary = boundary
        self.v0_decay = v0_decay
        self.v1_decay = v1_decay
        self.num_ = 0
        self.v0_to_v1_ = True
        self.from_ = self.v0
        self.to_ = self.v1

    def __call__(self, in_dict, previous_output_dict):
        progress = self.num_ / (self.frequency - 1)
        val = progress * (self.to_ - self.from_) + self.from_
        self.num_ += 1
        if self.num_ == self.frequency:
            if self.boundary == "wrap":
                self.num_ = 0
                self.v0 *= self.v0_decay
                self.v1 *= self.v1_decay
                self.from_ = self.v0
                self.to_ = self.v1
            elif self.boundary == "mirror":
                self.num_ = 1
                if self.v0_to_v1_:
                    self.v0 *= self.v0_decay
                    self.from_ = self.v1
                    self.to_ = self.v0
                else:
                    self.v1 *= self.v1_decay
                    self.from_ = self.v0
                    self.to_ = self.v1
                self.v0_to_v1_ = not self.v0_to_v1_
            else:
                raise ValueError("Unknown boundary: {}".format(self.boundary))
        return val
