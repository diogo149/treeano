import time

import six
import numpy as np

from .. import network_utils
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

    fmt:
    eg. "train_%s", "valid_%s"
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


class MonitorNetworkState(base.NetworkHandlerImpl):

    """
    handler that monitors shared variables in the network, storing them
    in an output map with the input format

    fmt:
    first %s corresponds to key, second corresponds to statistic
    eg. "network_%s_%s"
    """

    def __init__(self, fmt="network_%s_%s"):
        self.fmt = fmt

    def __call__(self, state, *args, **kwargs):
        res = super(MonitorNetworkState, self).__call__(state, *args, **kwargs)
        value_dict = network_utils.to_value_dict(state.network)
        for k, v in value_dict.items():
            abs_v = np.abs(v)
            res[self.fmt % (k, "abs->max")] = np.max(abs_v)
            # res[self.fmt % (k, "abs->mean")] = np.mean(abs_v)
            # res[self.fmt % (k, "abs->min")] = np.min(abs_v)
            res[self.fmt % (k, "mean")] = np.mean(v)
            res[self.fmt % (k, "std")] = np.std(v)
        return res

monitor_network_state = MonitorNetworkState


class MonitorVariable(base.NetworkHandlerImpl):

    """
    monitors a variable in a graph by raveling it into multiple scalars
    """

    def __init__(self, query, fmt="%s_%d"):
        self.query = query
        self.fmt = fmt

    def transform_compile_function_kwargs(self, state, **kwargs):
        if isinstance(self.query, six.string_types):
            node_name = self.query
            from_key = "default"
        elif isinstance(self.query, tuple):
            node_name, from_key = self.query
        else:
            assert False

        vw = state.network[node_name].get_variable(from_key)
        self.vw_name_ = vw.name
        self.output_key_ = self.fmt % (self.vw_name_, 0)
        assert self.output_key_ not in kwargs["outputs"]
        kwargs["outputs"][self.output_key_] = (node_name, from_key)
        return kwargs

    def call(self, fn, in_dict, *args, **kwargs):
        res = fn(in_dict, *args, **kwargs)
        val = res.pop(self.output_key_)
        for i, v in enumerate(val.ravel()):
            output_key = self.fmt % (self.vw_name_, i)
            assert output_key not in res
            res[output_key] = v
        return res

monitor_variable = MonitorVariable
