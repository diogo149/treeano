from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time


def _default_evaluate_until_callback(iter_num, res):
    print("Iter(%d): %s" % (iter_num, res))


def evaluate_until(fn,
                   gen,
                   max_iters=None,
                   max_seconds=None,
                   callback=_default_evaluate_until_callback):
    """
    evaluates a function on the output of a data generator until a given
    stopping condition
    """
    start_time = time.time()
    for i, data in enumerate(gen):
        if max_iters is not None and i >= max_iters:
            break
        if max_seconds is not None and max_seconds >= time.time() - start_time:
            break
        res = fn(data)
        if callback is not None:
            callback(i, res)
