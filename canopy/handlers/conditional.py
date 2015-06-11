from . import base


class CallAfterEvery(base.NetworkHandlerImpl):

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
