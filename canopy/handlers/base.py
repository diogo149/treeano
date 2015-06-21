from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import abc
import contextlib
import time

import six


class NetworkHandlerAPI(six.with_metaclass(abc.ABCMeta, object)):

    """
    base class for network handlers
    """

    def set_inner(self, inner_handler):
        self._inner_handler = inner_handler

    def build(self, state, input_network):
        # save a copy of the input network
        self._input_network = input_network
        self._build_from_input(state)

    def _build_from_input(self, state):
        # optionally change the input network
        output_network = self.transform_network(self._input_network)
        # recurse inward
        self._inner_handler.build(state, output_network)

    def compile_function(self, state, kwargs):
        # save a copy of original kwargs
        self._kwargs = kwargs
        self._compile_function_from_input(state)

    def _compile_function_from_input(self, state):
        # optionally change args/kwargs
        kwargs = self.transform_compile_function_kwargs(state, **self._kwargs)
        # make sure kwargs are not just a mutated version
        assert kwargs is not self._kwargs
        # recurse inward
        self._inner_handler.compile_function(state, kwargs)

    def initial_build(self, state, input_network, **kwargs):
        with state.time_build():
            self.build(state, input_network)
        with state.time_compile_function():
            self.compile_function(state, kwargs)

    def rebuild(self, state):
        with state.time_build():
            self._build_from_input(state)
        with state.time_build():
            self._compile_function_from_input(state)

    # ######################### methods to be overriten #######################

    @abc.abstractmethod
    def transform_network(self, network):
        pass

    @abc.abstractmethod
    def transform_compile_function_kwargs(self, state, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, state, *args, **kwargs):
        pass


class NetworkHandlerImpl(NetworkHandlerAPI):

    """
    network handler with default implementations
    """

    def transform_network(self, network):
        return network

    def transform_compile_function_kwargs(self, state, **kwargs):
        return kwargs

    def __call__(self, state, *args, **kwargs):
        """
        by default, redirect to NetworkHandlerAPI.call for a simpler API
        """

        def inner(*args, **kwargs):
            return self._inner_handler(state, *args, **kwargs)

        return self.call(inner, *args, **kwargs)

    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class FinalHandler(NetworkHandlerImpl):

    """
    ending network handler to actually call the function
    """

    def build(self, state, input_network):
        # save a copy of the input network
        self._input_network = input_network
        # update state
        state.update_network(input_network)

    def compile_function(self, state, kwargs):
        # save a copy of original kwargs
        self._kwargs = kwargs
        # actually compile function
        state.compile_function(kwargs)

    def __call__(self, state, *args, **kwargs):
        # call actual function with args, kwargs
        return state.call(*args, **kwargs)


class _HandledFunctionState(object):

    def __init__(self, initial_network):
        self.initial_network = initial_network

    def update_network(self, network):
        self.network = network
        if not self.network.is_built:
            self.network.build()

    def compile_function(self, kwargs):
        self.fn = self.network.function(**kwargs)

    def call(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @contextlib.contextmanager
    def time_build(self):
        # TODO add to total (keep state)
        start_time = time.time()
        yield
        total_time = time.time() - start_time
        print("Building took %fs" % total_time)

    @contextlib.contextmanager
    def time_compile_function(self):
        # TODO add to total (keep state)
        start_time = time.time()
        yield
        total_time = time.time() - start_time
        print("Compiling took %fs" % total_time)

    @contextlib.contextmanager
    def time_call(self):
        # TODO add to total (keep state)
        yield


class _HandledFunction(object):

    """
    class that stores handler-chain wide state
    """

    def __init__(self, network, handlers, inputs, outputs=None, **kwargs):
        self.network = network
        self.handlers = handlers + [FinalHandler()]

        self.state = _HandledFunctionState(network)

        for outer, inner in zip(self.handlers, self.handlers[1:]):
            outer.set_inner(inner)

        self.outermost = self.handlers[0]
        self.outermost.initial_build(self.state,
                                     self.network,
                                     inputs=inputs,
                                     outputs=outputs,
                                     **kwargs)

    def __call__(self, *args, **kwargs):
        return self.outermost(self.state, *args, **kwargs)

handled_function = _HandledFunction
