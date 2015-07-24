from . import base


class CallbackWithInput(base.NetworkHandlerImpl):

    """
    handler that calls a callback with both the result of the function and the
    inputs to the function
    """

    def __init__(self, callback):
        self.callback = callback

    def call(self, fn, in_map, *args, **kwargs):
        res = fn(in_map, *args, **kwargs)
        self.callback(res, in_map)
        return res

callback_with_input = CallbackWithInput
