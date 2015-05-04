from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import operator

import toolz


class UpdateDeltas(object):

    def __init__(self, deltas):
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
        return UpdateDeltas({k: fn(v) for k, v in self.deltas.items()})

    def __add__(self, other):
        if isinstance(other, UpdateDeltas):
            return UpdateDeltas(toolz.merge_with(sum,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: x + other)

    def __mul__(self, other):
        if isinstance(other, UpdateDeltas):
            def product(iterable):
                return reduce(operator.mul, iterable, 1)
            # TODO this will currently make it such that if one instance
            # has updates and another doesn't, it will return the same value
            # (another approach would be returning 0 if the value isn't in
            # both)
            # TODO is multiply by another set of deltas ever desired?
            return UpdateDeltas(toolz.merge_with(product,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: x * other)

    def __getitem__(self, key):
        return self.deltas.get(key, 0)

    def __setitem__(self, key, value):
        self.deltas[key] = value
