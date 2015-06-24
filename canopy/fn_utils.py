from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time


def _default_evaluate_until_callback(iter_num, res):
    print("Iter(%d): %s" % (iter_num, res))


# TODO move to handlers-specific module, since this assumes a handled_fn as
# input
def evaluate_until(fn,
                   gen,
                   max_iters=None,
                   max_seconds=None,
                   callback=_default_evaluate_until_callback):
    """
    evaluates a function on the output of a data generator until a given
    stopping condition

    fn:
    handled_fn
    """
    start_time = time.time()
    new_gen = enumerate(gen)
    try:
        while True:
            with fn.state.time("generating_data"):
                i, data = new_gen.next()
            if (max_iters is not None) and (i >= max_iters):
                break
            if ((max_seconds is not None)
                    and (time.time() - start_time >= max_seconds)):
                break
            res = fn(data)
            if callback is not None:
                callback(i, res)
    except StopIteration:
        pass
