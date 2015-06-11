from .base import NetworkHandlerImpl


class FunctionHandler(NetworkHandlerImpl):

    """
    handler that wraps functions and doesn't mutate the network
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state, *args, **kwargs):

        def inner(*args, **kwargs):
            return self._inner_handler(state, *args, **kwargs)

        res = self.fn(*args, **kwargs)
        return res


def function_handler(fn):
    """
    decorator for handlers which simply apply a function
    """
    return FunctionHandler(fn)
