from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import abc
import contextlib
import time
import collections

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
        with state.time("build"):
            self.build(state, input_network)
        with state.time("compile_function"):
            self.compile_function(state, kwargs)

    def rebuild(self, state):
        with state.time("build"):
            self._build_from_input(state)
        with state.time("build"):
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
        self.time_total = collections.defaultdict(lambda: 0)
        self.time_count = collections.defaultdict(lambda: 0)

    def update_network(self, network):
        self.network = network
        if not self.network.is_built:
            self.network.build()

    def compile_function(self, kwargs):
        with self.time("network_compile"):
            self.fn = self.network.function(**kwargs)

    def call(self, *args, **kwargs):
        with self.time("network_call"):
            return self.fn(*args, **kwargs)

    @contextlib.contextmanager
    def time(self, title):
        start_time = time.time()
        yield
        total_time = time.time() - start_time
        self.time_total[title] += total_time
        self.time_count[title] += 1
        # TODO figure out right way to print network info
        if title != "network_call":
            print("%s took %0.4fs" % (title, total_time))
