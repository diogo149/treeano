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


@treeano.register_node("hc_lstm")
class LSTMNode(treeano.NodeImpl):

    hyperparameter_names = ("inits",
                            "num_units",
                            "learn_init",
                            "backwards",
                            "grad_clip",
                            "only_return_final",
                            "forget_gate_bias")

    def compute_output(self, network, in_vw):
        num_units = network.find_hyperparameter(["num_units"])
        learn_init = network.find_hyperparameter(["learn_init"], False)
        backwards = network.find_hyperparameter(["backwards"], False)
        grad_clip = network.find_hyperparameter(["grad_clip"], None)
        forget_gate_bias = network.find_hyperparameter(["forget_gate_bias"], 0)
        only_return_final = network.find_hyperparameter(["only_return_final"],
                                                        False)

        # input should have axes (batch, time, features)
        assert in_vw.ndim == 3
        num_inputs = in_vw.shape[-1]
        assert num_inputs is not None

        def make_bias(name):
            return network.create_vw(
                name=name + "_bias",
                is_shared=True,
                shape=(num_units,),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable

        def make_weight(name, in_units):
            return network.create_vw(
                name=name + "_weight",
                is_shared=True,
                shape=(in_units, num_units),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

        def make_i_weight(name):
            return make_weight("i_to_%s" % name, num_inputs)

        def make_h_weight(name):
            return make_weight("h_to_%s" % name, num_units)

        i_to_update = make_i_weight("update")
        i_to_forgetgate = make_i_weight("forgetgate")
        i_to_inputgate = make_i_weight("inputgate")
        i_to_outputgate = make_i_weight("outputgate")
        h_to_update = make_h_weight("update")
        h_to_forgetgate = make_h_weight("forgetgate")
        h_to_inputgate = make_h_weight("inputgate")
        h_to_outputgate = make_h_weight("outputgate")
        update_bias = make_bias("update")
        forgetgate_bias = make_bias("forgetgate") + forget_gate_bias
        inputgate_bias = make_bias("inputgate")
        outputgate_bias = make_bias("outputgate")

        i_to_h = T.concatenate([i_to_update,
                                i_to_forgetgate,
                                i_to_inputgate,
                                i_to_outputgate], axis=1)
        nonsequences = [h_to_update,
                        h_to_forgetgate,
                        h_to_inputgate,
                        h_to_outputgate]
        h_to_h = T.concatenate(nonsequences, axis=1)
        b = T.concatenate([update_bias,
                           forgetgate_bias,
                           inputgate_bias,
                           outputgate_bias])

        def make_initial_state(name):
            if learn_init:
                initial_state = network.create_vw(
                    name=name,
                    is_shared=True,
                    shape=(num_units,),
                    tags = {"parameter", "bias"},
                    default_inits = [],
                ).variable
            else:
                initial_state = T.zeros((num_units,), dtype=fX)
            # repeat initial state for whole minibatch
            initial_state = T.repeat(initial_state.reshape((1, -1)),
                                     repeats=in_vw.symbolic_shape()[0],
                                     axis=0)
            return initial_state

        initial_cell_state = make_initial_state("initial_cell_state")
        initial_hidden_state = make_initial_state("initial_hidden_state")

        in_var = in_vw.variable
        # precompute i to h and bias
        in_feats = T.dot(in_var, i_to_h) + b.dimshuffle("x", "x", 0)
        # convert axes to order (time, batch, features)
        in_feats = in_feats.dimshuffle(1, 0, 2)

        def step(in_feat, c_prev, h_prev, *_nonsequences):
            logit = in_feat + T.dot(h_prev, h_to_h)
            if grad_clip is not None:
                logit = theano.gradient.grad_clip(logit, -grad_clip, grad_clip)

            update_logit = logit[:, :num_units]
            forget_logit = logit[:, num_units:num_units * 2]
            input_logit = logit[:, num_units * 2:num_units * 3]
            output_logit = logit[:, num_units * 3:]

            update = T.tanh(update_logit)
            forget_gate = T.nnet.sigmoid(forget_logit)
            input_gate = T.nnet.sigmoid(input_logit)
            output_gate = T.nnet.sigmoid(output_logit)

            c = input_gate * update + forget_gate * c_prev
            h = output_gate * T.tanh(c)
            return c, h

        res, _updates = theano.scan(fn=step,
                                    sequences=[in_feats],
                                    outputs_info=[initial_cell_state,
                                                  initial_hidden_state],
                                    non_sequences=nonsequences,
                                    go_backwards=backwards,
                                    strict=True)

        cells, hidden_states = res
        out_var = hidden_states

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


@treeano.register_node("hc_gru")
class GRUNode(treeano.NodeImpl):

    hyperparameter_names = ("inits",
                            "num_units",
                            "learn_init",
                            "backwards",
                            "grad_clip",
                            "only_return_final",
                            "update_gate_bias")

    def compute_output(self, network, in_vw):
        num_units = network.find_hyperparameter(["num_units"])
        learn_init = network.find_hyperparameter(["learn_init"], False)
        backwards = network.find_hyperparameter(["backwards"], False)
        grad_clip = network.find_hyperparameter(["grad_clip"], None)
        update_gate_bias = network.find_hyperparameter(["update_gate_bias"], 0)
        only_return_final = network.find_hyperparameter(["only_return_final"],
                                                        False)

        # input should have axes (batch, time, features)
        assert in_vw.ndim == 3
        num_inputs = in_vw.shape[-1]
        assert num_inputs is not None

        def make_bias(name):
            return network.create_vw(
                name=name + "_bias",
                is_shared=True,
                shape=(num_units,),
                tags={"parameter", "bias"},
                default_inits=[],
            ).variable

        def make_weight(name, in_units):
            return network.create_vw(
                name=name + "_weight",
                is_shared=True,
                shape=(in_units, num_units),
                tags={"parameter", "weight"},
                default_inits=[],
            ).variable

        def make_i_weight(name):
            return make_weight("i_to_%s" % name, num_inputs)

        def make_h_weight(name):
            return make_weight("h_to_%s" % name, num_units)

        i_to_updategate = make_i_weight("updategate")
        i_to_resetgate = make_i_weight("resetgate")
        i_to_update = make_i_weight("update")
        h_to_updategate = make_h_weight("updategate")
        h_to_resetgate = make_h_weight("resetgate")
        h_to_update = make_h_weight("update")
        updategate_bias = make_bias("updategate") + update_gate_bias
        resetgate_bias = make_bias("resetgate")
        update_bias = make_bias("update")

        i_to_h = T.concatenate([i_to_updategate,
                                i_to_resetgate,
                                i_to_update], axis=1)
        nonsequences = [h_to_updategate,
                        h_to_resetgate,
                        h_to_update]
        b = T.concatenate([updategate_bias,
                           resetgate_bias,
                           update_bias])

        def make_initial_state(name):
            if learn_init:
                initial_state = network.create_vw(
                    name=name,
                    is_shared=True,
                    shape=(num_units,),
                    tags = {"parameter", "bias"},
                    default_inits = [],
                ).variable
            else:
                initial_state = T.zeros((num_units,), dtype=fX)
            # repeat initial state for whole minibatch
            initial_state = T.repeat(initial_state.reshape((1, -1)),
                                     repeats=in_vw.symbolic_shape()[0],
                                     axis=0)
            return initial_state

        initial_state = make_initial_state("initial_state")

        in_var = in_vw.variable
        # precompute i to h and bias
        in_feats = T.dot(in_var, i_to_h) + b.dimshuffle("x", "x", 0)
        # convert axes to order (time, batch, features)
        in_feats = in_feats.dimshuffle(1, 0, 2)

        def slice_gates(x, gate_idx):
            return x[:, gate_idx * num_units:(gate_idx + 1) * num_units]

        def step(in_feat, h_prev, *_nonsequences):
            h_feat = T.dot(h_prev, T.concatenate([h_to_updategate,
                                                  h_to_resetgate],
                                                 axis=1))

            updategate_logit = slice_gates(in_feat, 0) + slice_gates(h_feat, 0)
            update_gate = T.nnet.sigmoid(updategate_logit)
            resetgate_logit = slice_gates(in_feat, 1) + slice_gates(h_feat, 1)
            reset_gate = T.nnet.sigmoid(resetgate_logit)

            update_logit = slice_gates(in_feat, 2) + T.dot(reset_gate * h_prev,
                                                           h_to_update)
            if grad_clip is not None:
                update_logit = theano.gradient.grad_clip(update_logit,
                                                         -grad_clip,
                                                         grad_clip)

            update = T.tanh(update_logit)

            h = (1 - update_gate) * h_prev + update_gate * update
            return h

        out_var, _updates = theano.scan(fn=step,
                                        sequences=[in_feats],
                                        outputs_info=[initial_state],
                                        non_sequences=nonsequences,
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
