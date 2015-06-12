from . import base


class CallAfterEvery(base.NetworkHandlerImpl):

    """
    handler that calls a callback with the result of a function every few
    calls
    """

    def __init__(self, frequency, callback):
        self.frequency = frequency
        self.callback = callback
        self.count = 0

    def call(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        self.count += 1
        if (self.count % self.frequency) == 0:
            self.callback(res)
        return res

call_after_every = CallAfterEvery
