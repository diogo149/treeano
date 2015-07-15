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
                   callback=_default_evaluate_until_callback,
                   catch_keyboard_interrupt=True):
    """
    evaluates a function on the output of a data generator until a given
    stopping condition

    fn:
    handled_fn
    """
    start_time = time.time()
    new_gen = enumerate(gen)

    to_catch = (StopIteration,)
    if catch_keyboard_interrupt:
        to_catch = to_catch + (KeyboardInterrupt,)

    print("Beginning evaluate_until")
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
    except to_catch:
        print("Ending evaluate_until")
