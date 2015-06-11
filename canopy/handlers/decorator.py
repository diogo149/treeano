from .base import NetworkHandlerImpl


class CallHandler(NetworkHandlerImpl):

    """
    handler that wraps function calls and doesn't mutate the network
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state, *args, **kwargs):

        def inner(*args, **kwargs):
            return self._inner_handler(state, *args, **kwargs)

        res = self.fn(inner, *args, **kwargs)
        return res


def call_handler(fn):
    """
    decorator for handlers which simply apply a function
    """
    return CallHandler(fn)
