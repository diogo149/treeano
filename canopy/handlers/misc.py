import numpy as np

import treeano

from .. import network_utils
from . import base


class CallbackWithInput(base.NetworkHandlerImpl):

    """
    handler that calls a callback with both the result of the function and the
    inputs to the function
    """

    def __init__(self, callback):
        self.callback = callback

    def call(self, fn, in_dict, *args, **kwargs):
        res = fn(in_dict, *args, **kwargs)
        self.callback(in_dict, res)
        return res

callback_with_input = CallbackWithInput


class ExponentialPolyakAveraging(base.NetworkHandlerImpl):

    """
    Polyak-Ruppert averaging using an exponential moving average

    handler that averages together weights over time

    see "Adam: A Method for Stochastic Optimization" v8 -> extensions ->
    temporal averaging
    (http://arxiv.org/abs/1412.6980)
    """

    def __init__(self, beta=0.9, average_only_floats=True):
        assert 0 < beta < 1
        self.beta = beta
        self.average_only_floats = average_only_floats
        self.iters_ = 0
        self.theta_bar_ = None

    def get_value_dict(self):
        value_dict = {}
        # unbias the estimates
        for k, v in self.theta_bar_.items():
            unbiased = (v / (1 - self.beta ** self.iters_)).astype(v.dtype)
            value_dict[k] = unbiased
        return value_dict

    def __call__(self, state, *args, **kwargs):
        res = super(ExponentialPolyakAveraging, self).__call__(
            state, *args, **kwargs)
        value_dict = network_utils.to_value_dict(state.network)
        self.iters_ += 1
        # initialize moving weights
        if self.theta_bar_ is None:
            self.theta_bar_ = {}
            for k, v in value_dict.items():
                self.theta_bar_[k] = np.zeros_like(v)
        # update values
        for k, v in value_dict.items():
            if (self.average_only_floats
                    and not treeano.utils.is_float_array(v)):
                # keep last
                # ---
                # because we might only want to save parameters
                # ie. not averaging things like batch counts
                self.theta_bar_[k] = v
            else:
                # exponential moving average
                prev = self.theta_bar_[k]
                curr = self.beta * prev + (1 - self.beta) * v
                self.theta_bar_[k] = curr.astype(prev.dtype)
        return res

exponential_polyak_averaging = ExponentialPolyakAveraging
