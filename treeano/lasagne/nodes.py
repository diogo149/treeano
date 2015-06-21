import toolz
import numpy as np
import lasagne

from .. import utils
from .. import core
from .. import nodes


def wrap_lasagne_node(network, in_vw, param_kwargs, constructor, kwargs):
    """
    param_kwargs:
    dict from param name to map of keyword arguments for constructing
    (eg. inits, tags, etc.)
    """
    l_in = lasagne.layers.InputLayer(
        input_var=in_vw.variable,
        shape=in_vw.shape)
    l_out = constructor(l_in, **kwargs)
    output = lasagne.layers.get_output(l_out)
    output_shape = lasagne.layers.get_output_shape(l_out)
    params = lasagne.layers.get_all_params(l_out)
    to_replace = {}
    for param in params:
        name = param.name
        assert name in param_kwargs
        assert "tags" in param_kwargs[name]
        assert "inits" in param_kwargs[name]
        vw = network.create_variable(
            name=name,
            is_shared=True,
            shape=param.get_value().shape,
            **param_kwargs[name]
        )
        to_replace[param] = vw.variable
    new_output, = utils.deep_clone([output], to_replace)
    network.create_variable(
        name="default",
        variable=new_output,
        shape=output_shape,
        tags={"output"},
    )


@core.register_node("lasagne_dense")
class DenseNode(core.NodeImpl):

    """
    node wrapping lasagne's DenseLayer
    """

    hyperparameter_names = ("inits",
                            "dense_num_units",
                            "num_units")

    def compute_output(self, network, in_vw):
        inits = list(toolz.concat(network.find_hyperparameters(
            ["inits"],
            [])))
        num_units = network.find_hyperparameter(["dense_num_units",
                                                 "num_units"])
        wrap_lasagne_node(
            network=network,
            in_vw=in_vw,
            param_kwargs=dict(
                W=dict(tags={"parameter", "weight"},
                       inits=inits),
                b=dict(tags={"parameter", "bias"},
                       inits=inits)
            ),
            constructor=lasagne.layers.DenseLayer,
            kwargs=dict(
                num_units=num_units,
                nonlinearity=lasagne.nonlinearities.identity,
            )
        )


def ReLUNode(name):
    return nodes.ApplyNode(name,
                           fn=lasagne.nonlinearities.rectify,
                           shape_fn=utils.identity)


@core.register_node("lasagne_sgd")
class SGDNode(core.Wrapper1NodeImpl):

    """
    node that provides updates via SGD
    """

    hyperparameter_names = ("cost_reference",
                            "reference",
                            "sgd_learning_rate",
                            "learning_rate")

    def new_update_deltas(self, network):
        cost_reference = network.find_hyperparameter(["cost_reference",
                                                      "reference"])
        cost = network[cost_reference].get_variable("default").variable
        parameters = network.find_vws_in_subtree(tags=["parameter"])
        learning_rate = network.find_hyperparameter(["sgd_learning_rate",
                                                     "learning_rate"])
        updates = lasagne.updates.sgd(cost,
                                      [parameter.variable
                                       for parameter in parameters],
                                      learning_rate)
        return core.UpdateDeltas.from_updates(updates)
