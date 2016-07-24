"""
from "Deep Inside Convolutional Networks: Visualising Image Classification
Models and Saliency Maps"
http://arxiv.org/abs/1312.6034
"""

import theano
import theano.tensor as T
import treeano
import canopy


class SensitivityAnalysisOutput(canopy.handlers.NetworkHandlerImpl):

    """
    adds a new input and output to the network
    - the input is an int that is an index into the logit
    - the output is a tensor of the same shape as the input representing
      the result of the sensitivity analysis

    idx_input_key: key of the index

    output_key: key to put the sensitivity analysis in the results

    input_name: node name of the input in the network

    logit_name: node name of the logit in the network
    """

    def __init__(self, idx_input_key, output_key, input_name, logit_name):
        self.idx_input_key = idx_input_key
        self.output_key = output_key
        self.input_name = input_name
        self.logit_name = logit_name

    def transform_compile_function_kwargs(self, state, **kwargs):
        assert self.idx_input_key not in kwargs["inputs"]
        assert self.output_key not in kwargs["outputs"]
        network = state.network
        input_var = network[self.input_name].get_vw("default").variable
        logit_var = network[self.logit_name].get_vw("default").variable
        assert logit_var.ndim == 2
        idx_var = T.iscalar()
        target_var = logit_var[:, idx_var].sum()
        sensitivity_var = T.grad(target_var, input_var)
        kwargs["inputs"][self.idx_input_key] = idx_var
        kwargs["outputs"][self.output_key] = sensitivity_var
        return kwargs


def sensitivity_analysis_fn(input_name,
                            logit_name,
                            network,
                            handlers,
                            inputs=None,
                            **kwargs):
    """
    returns a function from input to sensitivity analysis heatmap
    """
    handlers = [
        SensitivityAnalysisOutput(idx_input_key="idx",
                                  output_key="outputs",
                                  input_name=input_name,
                                  logit_name=logit_name),
        canopy.handlers.override_hyperparameters(deterministic=True)
    ] + handlers

    fn = canopy.handled_fn(network,
                           handlers=handlers,
                           inputs={"input": input_name},
                           outputs={},
                           **kwargs)

    def inner(in_val, idx_val):
        return fn({"input": in_val, "idx": idx_val})["outputs"]

    return inner


def customizable_sensitivity_analysis_fn(input_name,
                                         logit_name,
                                         network,
                                         handlers,
                                         inputs,
                                         outputs=None,
                                         **kwargs):
    """
    returns a function from input to sensitivity analysis heatmap

    takes in additional keys for "input" and "idx"
    """
    if outputs is None:
        outputs = {}

    assert "outputs" not in outputs

    handlers = [
        SensitivityAnalysisOutput(idx_input_key="idx",
                                  output_key="outputs",
                                  input_name=input_name,
                                  logit_name=logit_name),
        canopy.handlers.override_hyperparameters(deterministic=True)
    ] + handlers

    assert "input" not in inputs
    assert "idx" not in inputs

    # make a copy of inputs so that we can mutate
    inputs = dict(inputs)
    inputs["input"] = input_name
    fn = canopy.handled_fn(network,
                           handlers=handlers,
                           inputs=inputs,
                           outputs=outputs,
                           **kwargs)

    return fn
