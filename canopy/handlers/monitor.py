from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time

import numpy as np

import treeano.theano_extensions.nanguardmode

from . import base


class TimeCall(base.NetworkHandlerImpl):

    """
    handler that times the inner handler call and adds a corresponding key
    with the time to the output time
    """

    def __init__(self, key="time"):
        self.key = key

    def call(self, fn, *args, **kwargs):
        start_time = time.time()
        res = fn(*args, **kwargs)
        total_time = time.time() - start_time
        assert self.key not in res
        res[self.key] = total_time
        return res

time_call = TimeCall


class TimePerRow(base.NetworkHandlerImpl):

    """
    handler that times the inner handler call, and estimates the time it
    takes for each row of input
    """

    def __init__(self, input_key, key="ms_per_row"):
        self.input_key = input_key
        self.key = key

    def call(self, fn, in_dict, *args, **kwargs):
        start_time = time.time()
        num_rows = len(in_dict[self.input_key])
        res = fn(in_dict, *args, **kwargs)
        total_time = time.time() - start_time
        assert self.key not in res
        res[self.key] = total_time / num_rows
        return res

time_per_row = TimePerRow


class EvaluateMonitoringVariables(base.NetworkHandlerImpl):

    """
    handler that additionally evaluates all monitoring variables, storing them
    in an output map with the input format
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def transform_compile_function_kwargs(self, state, **kwargs):
        network = state.network.relative_network()
        vws = network.find_vws_in_subtree(tags={"monitor"})
        for vw in vws:
            name = self.fmt % vw.name
            assert name not in kwargs["outputs"]
            kwargs["outputs"][name] = vw.variable
        return kwargs

evaluate_monitoring_variables = EvaluateMonitoringVariables


class OutputNanGuard(base.NetworkHandlerImpl):

    """
    handler that checks outputs for nan and raises an exception if any
    contain nan

    should be more efficient than theano.compile.nanguardmode.NanGuardMode,
    since it is only done on outputs and not intermediate computations
    """

    def __init__(self,
                 nan_is_error=True,
                 inf_is_error=True,
                 big_is_error=True,
                 action="error"):
        self.nan_is_error = nan_is_error
        self.inf_is_error = inf_is_error
        self.big_is_error = big_is_error
        self.action = action

    def _handle_error(self, error_type, k, v):
        msg = dict(
            msg="OutputNanGuard error found!",
            error_type=error_type,
            key=k,
            value=v
        )
        if self.action == "error":
            raise Exception(msg)
        elif self.action == "print":
            print(msg)
        else:
            raise ValueError("incorrect action: %s" % self.action)

    def call(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        for k, v in res.items():
            if self.nan_is_error:
                if np.any(np.isnan(v)):
                    self._handle_error("nan", k, v)
            if self.inf_is_error:
                if np.any(np.isinf(v)):
                    self._handle_error("inf", k, v)
            if self.big_is_error:
                if np.any(np.abs(v) > 1e10):
                    self._handle_error("big", k, v)
        return res

output_nanguard = OutputNanGuard


class NanGuardMode(base.NetworkHandlerImpl):

    """
    handler that changes the mode to theano.compile.nanguardmode.NanGuardMode

    http://deeplearning.net/software/theano/library/compile/nanguardmode.html
    """

    def __init__(self,
                 nan_is_error=True,
                 inf_is_error=True,
                 big_is_error=True):
        self.nan_is_error = nan_is_error
        self.inf_is_error = inf_is_error
        self.big_is_error = big_is_error

    def transform_compile_function_kwargs(self, state, **kwargs):
        # don't overwrite an existing mode
        assert "mode" not in kwargs
        kwargs["mode"] = treeano.theano_extensions.nanguardmode.NanGuardMode(
            nan_is_error=self.nan_is_error,
            inf_is_error=self.inf_is_error,
            big_is_error=self.big_is_error
        )
        return kwargs

nanguardmode = NanGuardMode

