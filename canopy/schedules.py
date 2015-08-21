"""
schedules for canopy.handlers.schedule_hyperparameter
"""
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


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
        # make a copy since we will mutate it
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
