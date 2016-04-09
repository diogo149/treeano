"""
for hard-coded recurrent nodes
"""

import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def bidirectional(name, node_constructor, axis=2, **kwargs):
    """
    eg.
    bidirectional("birnn", RNNNode, num_units=32)
    """
    return tn.ConcatenateNode(
        name,
        [node_constructor(name + "_forward", backwards=False, **kwargs),
         node_constructor(name + "_backward", backwards=True, **kwargs)],
        axis=axis)


@treeano.register_node("hc_rnn")
class RNNNode(treeano.NodeImpl):

    """
    "vanilla" RNN with dense hidden to hidden connections
    """

    hyperparameter_names = ("inits",
                            "num_units",
                            "learn_init",
                            "backwards",
                            "grad_clip",
                            "only_return_final",)

    def compute_output(self, network, in_vw):
        num_units = network.find_hyperparameter(["num_units"])
        learn_init = network.find_hyperparameter(["learn_init"], False)
        backwards = network.find_hyperparameter(["backwards"], False)
        grad_clip = network.find_hyperparameter(["grad_clip"], None)
        only_return_final = network.find_hyperparameter(["only_return_final"],
                                                        False)

        # input should have axes (batch, time, features)
        assert in_vw.ndim == 3
        num_inputs = in_vw.shape[-1]
        assert num_inputs is not None

        i_to_h_weight = network.create_vw(
            name="i_to_h_weight",
            is_shared=True,
            shape=(num_inputs, num_units),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable
        h_to_h_weight = network.create_vw(
            name="h_to_h_weight",
            is_shared=True,
            shape=(num_units, num_units),
            tags={"parameter", "weight"},
            default_inits=[],
        ).variable
        b = network.create_vw(
            name="bias",
            is_shared=True,
            shape=(num_units,),
            tags={"parameter", "bias"},
            default_inits=[],
        ).variable
        if learn_init:
            initial_state = network.create_vw(
                name="initial_state",
                is_shared=True,
                shape=(num_units,),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable
        else:
            initial_state = T.zeros((num_units,), dtype=fX)
        # repeat initial state for whole minibatch
        initial_state = T.repeat(initial_state.reshape((1, -1)),
                                 repeats=in_vw.symbolic_shape()[0],
                                 axis=0)

        in_var = in_vw.variable
        # precompute i to h and bias
        in_feats = T.dot(in_var, i_to_h_weight) + b.dimshuffle("x", "x", 0)
        # convert axes to order (time, batch, features)
        in_feats = in_feats.dimshuffle(1, 0, 2)

        def step(in_feat, h_prev, _h_to_h_weight):
            logit = in_feat + T.dot(h_prev, h_to_h_weight)
            if grad_clip is not None:
                logit = theano.gradient.grad_clip(logit, -grad_clip, grad_clip)
            h = T.tanh(logit)
            return h

        out_var, _updates = theano.scan(fn=step,
                                        sequences=[in_feats],
                                        outputs_info=[initial_state],
                                        non_sequences=[h_to_h_weight],
                                        go_backwards=backwards,
                                        strict=True)

        if only_return_final:
            out_var = out_var[-1]

            out_shape = (in_vw.shape[0], num_units)

            network.create_vw(
                name="default",
                variable=out_var,
                shape=out_shape,
                tags={"output"},
            )
        else:
            # convert axes back to (batch, time, features)
            out_var = out_var.dimshuffle(1, 0, 2)
            if backwards:
                out_var = out_var[:, ::-1]

            out_shape = in_vw.shape[:-1] + (num_units,)

            network.create_vw(
                name="default",
                variable=out_var,
                shape=out_shape,
                tags={"output"},
            )
