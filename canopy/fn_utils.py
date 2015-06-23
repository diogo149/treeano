import time


def evaluate_until(fn, gen, max_iters=None, max_seconds=None, callback=None):
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
            callback(res)
