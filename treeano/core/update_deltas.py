from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import operator
import functools

import toolz


def _smart_reduce(op, iterable):
    iterable = list(iterable)
    if len(iterable) == 1:
        return iterable[0]
    else:
        return reduce(op, iterable[1:], iterable[0])


def _smart_add(x, y):
    """
    0-aware add, to prevent computation graph from getting very large
    """
    if x == 0:
        return y
    elif y == 0:
        return x
    else:
        return x + y


def _smart_mul(x, y):
    """
    0-aware multiply, to prevent computation graph from getting very large
    """
    if x == 0 or y == 0:
        return 0
    elif x == 1:
        return y
    elif y == 1:
        return x
    else:
        return x * y

_sum = functools.partial(_smart_reduce, _smart_add)
_product = functools.partial(_smart_reduce, _smart_mul)


class UpdateDeltas(object):

    def __init__(self, deltas=None):
        if deltas is None:
            deltas = {}
        assert isinstance(deltas, dict)
        self.deltas = deltas

    def to_updates(self):
        updates = []
        for var, delta in self.deltas.items():
            updates.append((var, var + delta))
        # sorting updates by name so that the order is deterministic
        updates.sort(key=lambda pair: pair[0].name)
        return updates

    @classmethod
    def from_updates(cls, updates):
        if isinstance(updates, list):
            delta_dict = {var: (new_value - var)
                          for var, new_value in updates}
        elif isinstance(updates, dict):
            delta_dict = {var: (new_value - var)
                          for var, new_value in updates.items()}
        else:
            raise ValueError("Can't handle updates of the given type")
        return cls(delta_dict)

    def apply(self, fn):
        """
        applies a function to all contained deltas and returns a new
        UpdateDeltas instance
        """
        return UpdateDeltas({k: fn(v) for k, v in self.deltas.items()})

    def iapply(self, fn):
        """
        applies a function to all contained deltas in place
        """
        self.deltas = {k: fn(v) for k, v in self.deltas.items()}
        return self

    def __add__(self, other):
        """
        adds a value, and returns a new instance of UpdateDeltas
        """
        if isinstance(other, UpdateDeltas):
            return UpdateDeltas(toolz.merge_with(_sum,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: _smart_add(x, other))

    def __iadd__(self, other):
        """
        mutates the UpdateDeltas by adding a value
        """
        if isinstance(other, UpdateDeltas):
            self.deltas = toolz.merge_with(_sum,
                                           self.deltas,
                                           other.deltas)
        else:
            self.iapply(lambda x: _smart_add(x, other))
        return self

    def __mul__(self, other):
        """
        multiplies a value, and returns a new instance of UpdateDeltas
        """
        if isinstance(other, UpdateDeltas):
            # TODO this will currently make it such that if one instance
            # has updates and another doesn't, it will return the same value
            # (another approach would be returning 0 if the value isn't in
            # both)
            # TODO is multiply by another set of deltas ever desired?
            return UpdateDeltas(toolz.merge_with(_product,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: _smart_mul(x, other))

    def __imul__(self, other):
        """
        mutates the UpdateDeltas by multiplying a value
        adds a value, and returns a new instance of UpdateDeltas
        """
        if isinstance(other, UpdateDeltas):
            self.deltas = toolz.merge_with(_product,
                                           self.deltas,
                                           other.deltas)
        else:
            self.iapply(lambda x: _smart_mul(x, other))
        return self

    def __getitem__(self, key):
        return self.deltas.get(key, 0)

    def __setitem__(self, key, value):
        self.deltas[key] = value

    def __contains__(self, item):
        return item in self.deltas
