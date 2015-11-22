"""
nodes for scan operations
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import theano

from .. import core
from .. import utils


@core.register_node("scan_input")
class ScanInputNode(core.NodeImpl):

    """
    Node that transfroms a sequence-wise input to an element-wise input,
    for replacement in a ScanNode.
    """

    def compute_output(self, network, in_vw):
        scan_axis = network.find_hyperparameter(["scan_axis"], 1)

        def remove_scan_axis(in_val):
            out_val = list(in_val)
            out_val.pop(scan_axis)
            # assuming in_val is a list or tuple
            if isinstance(in_val, tuple):
                return tuple(out_val)
            else:
                assert isinstance(in_val, list)
                return out_val

        # construct output
        network.create_vw(
            name="default",
            is_shared=False,
            shape=remove_scan_axis(in_vw.shape),
            broadcastable=remove_scan_axis(in_vw.broadcastable),
            tags={"input"},
        )


@core.register_node("scan_state")
class ScanStateNode(core.NodeImpl):

    """
    Container for hidden state, where one specifies an initial value and
    a next value via other nodes in the tree.

    NOTE: initial state MUST have the correct shape (ie. same shape as
    the output of the next_state node)

    initial_scan_state_reference:
    optional reference to a node to take initial state from
    if not given, the default input of the node is used
    """

    hyperparameter_names = ("initial_scan_state_reference",
                            "initial_state_reference",
                            "initial_state",
                            "next_scan_state_reference",
                            "next_state_reference",
                            "next_state")
    input_keys = ("initial_state",)

    def init_state(self, network):
        node_name = network.find_hyperparameter(
            ["initial_scan_state_reference",
             "initial_state_reference",
             "initial_state"],
            None)
        if node_name is None:
            # otherwise set it to the default input of the node
            node_name = network.get_all_input_edges()["default"]
        # add dependency in dag
        network.take_output_from(
            node_name,
            to_key="initial_state")
        # set next state node
        next_state = network.find_hyperparameter(["next_scan_state_reference",
                                                  "next_state_reference",
                                                  "next_state"])
        network.set_data("next_state", next_state)

    def compute_output(self, network, initial_state):
        # copy initial state, to be later referenced
        network.copy_vw(
            name="initial_state",
            previous_vw=initial_state,
            tags={"input"},
        )
        # create a new variable representing the output of this node,
        # so that the scan node can replace it with the node's input at a
        # previous time step
        network.create_vw(
            name="default",
            is_shared=False,
            shape=initial_state.shape,
            broadcastable=initial_state.broadcastable,
            tags={"output"},
        )


@core.register_node("scan")
class ScanNode(core.Wrapper1NodeImpl):

    """
    root node for a scan operation. transforms an element-wise child subtree
    into a sequence-wise subtree, taking into account ScanStateNode's
    """

    hyperparameter_names = ("scan_axis", )
    input_keys = ("default", "final_child_output",)

    def __init__(self, name, *args, **kwargs):
        super(ScanNode, self).__init__(name, *args, **kwargs)
        self._scan_input_node = ScanInputNode(name=self.name + "_input")

    def architecture_children(self):
        return ([self._scan_input_node]
                + super(ScanNode, self).architecture_children())

    def init_state(self, network):
        # send input to scan input node, and take output from child
        super(ScanNode, self).init_state(network)
        # link scan input node to child
        child1, child2 = self.architecture_children()
        network.add_dependency(child1.name, child2.name)

    def compute_output(self, network, sequence_input, element_output):
        scan_axis = network.find_hyperparameter(["scan_axis"], 1)
        # FIXME delete hard coded axis when actually shuffling dimensions
        scan_axis = 0

        # ################################ sequences ##########################

        # for now, only the single ScanInputNode
        input_network = network[self._scan_input_node.name]
        element_input = input_network.get_vw("default").variable
        element_input_vars = [element_input]
        # FIXME shuffle dimensions to use scan_axis
        # TODO assert ndim is right
        input_sequences = [sequence_input.variable]

        # ############################### outputs_info #######################
        # ---
        # sources:
        # 1. all outputs of the subtree
        # 2. next state for scan state node

        # outputs
        # original_outputs are ones whose variable will be replaced with a
        # version with an additional scan axis
        # NOTE: having  variables from nested scans is desired here, because
        # we want to transform the output into a version with a sequence
        # dimension
        original_outputs = network.find_vws_in_subtree(tags={"output"})
        # set all_outputs to a copy of original_outputs
        all_outputs = list(original_outputs)
        # in case the element-wise output of the scan is not tagged as an
        # "output" variable, we don't want to replace it with a sequence
        if element_output not in all_outputs:
            all_outputs.append(element_output)
        # find corresponding variable wrappers
        element_output_vars = [variable_wrapper.variable
                               for variable_wrapper in all_outputs]
        # set initial outputs_info that we do not care about the values
        outputs_info = [None] * len(all_outputs)

        # scan state nodes
        # FIXME do not get state in a nested scan
        scan_state_nodes = network.find_nodes_in_subtree(ScanStateNode)
        scan_state_networks = [network[node.name] for node in scan_state_nodes]
        scan_state_outputs = [net.get_vw("default")
                              for net in scan_state_networks]
        scan_state_vars = [variable_wrapper.variable
                           for variable_wrapper in scan_state_outputs]
        # finding idxs of scan state nodes in
        scan_state_idxs = [all_outputs.index(var)
                           for var in scan_state_outputs]
        # find the order that the states will appear in scan's inputs
        scan_state_order = sorted(range(len(scan_state_idxs)),
                                  key=lambda x: scan_state_idxs[x])
        # get the node name of the next state from each scan state node
        scan_state_next_names = [net.get_data("next_state")
                                 for net in scan_state_networks]
        # each scan state should probably have a unique next state
        # delete the assertion if this assumption does not hold
        # (the code should work just fine, this is just a sanity check)
        assert len(scan_state_next_names) == len(set(scan_state_next_names))
        scan_state_next_networks = [network[name]
                                    for name in scan_state_next_names]
        scan_state_next_vws = [net.get_vw("default")
                               for net in scan_state_next_networks]
        scan_state_next_idxs = [all_outputs.index(var)
                                for var in scan_state_next_vws]
        # finding initial states
        scan_state_initial_vws = [net.get_vw("initial_state")
                                  for net in scan_state_networks]
        # updates outputs_info to contain initial state
        for idx, node, init_vw, next_vw in zip(scan_state_idxs,
                                               scan_state_nodes,
                                               scan_state_initial_vws,
                                               scan_state_next_vws):
            # make sure initial and final shape are the same
            # ---
            # NOTE: node is only passed in for debugging purposes
            assert init_vw.shape == next_vw.shape, dict(
                msg=("Initial and final state from ScanStateNode must be "
                     "the same."),
                node=node,
                init_shape=init_vw.shape,
                next_shape=next_vw.shape,
                init_vw=init_vw,
                next_vw=next_vw,
            )
            outputs_info[idx] = init_vw.variable

        # ############################## non_sequences ########################
        # ---
        # for now, don't specify any non-sequences and hope that theano
        # can optimize the graph
        non_sequence_vars = []
        non_sequences = []

        # ################################### scan ############################

        def step(*scan_vars):
            # calculate number for each type of scan var
            num_inputs = len(input_sequences)
            num_outputs = len([x for x in outputs_info if x is not None])
            num_non_sequences = len(non_sequences)
            assert len(scan_vars) == (num_inputs
                                      + num_outputs
                                      + num_non_sequences)
            assert num_outputs == len(scan_state_nodes)

            # break down scan vars into appropriate categories
            scan_input_vars = scan_vars[:num_inputs]
            scan_output_vars = scan_vars[num_inputs:num_inputs + num_outputs]
            scan_non_sequences = scan_vars[num_inputs + num_outputs:]

            # setup variables for replacement
            to_replace = []
            for_replace = []
            # input vars
            to_replace += element_input_vars
            for_replace += scan_input_vars
            # non sequences
            to_replace += non_sequence_vars
            for_replace += scan_non_sequences
            # scan state nodes
            for state_idx, scan_output_var in zip(scan_state_order,
                                                  scan_output_vars):
                to_replace.append(scan_state_vars[state_idx])
                for_replace.append(scan_output_var)

            assert len(to_replace) == len(for_replace)

            # perform scan
            new_outputs = utils.deep_clone(
                element_output_vars,
                replace=dict(zip(to_replace, for_replace)),
            )

            final_outputs = list(new_outputs)

            # set next state for recurrent state nodes
            for state_idx, next_idx in zip(scan_state_idxs,
                                           scan_state_next_idxs):
                final_outputs[state_idx] = final_outputs[next_idx]

            return final_outputs

        # edit all outputs of subtree
        results, updates = theano.scan(
            fn=step,
            outputs_info=outputs_info,
            sequences=input_sequences,
            non_sequences=non_sequences,
        )

        # ############################# post-processing #######################

        # scan automatically unwraps lists, so rewrap if needed
        if not isinstance(results, list):
            results = [results]
        assert len(results) == len(element_output_vars)
        result_map = dict(zip(element_output_vars, results))

        # TODO store updates in network (to later be used in new_update_deltas)
        # NOTE: before doing this, look into the effects of manipulating the
        # update deltas of random variables - it might not work as expected

        # FIXME unshuffle dimensions for scan axes

        def transform_shape(old_shape):
            tmp = list(old_shape)
            tmp.insert(scan_axis, None)
            # FIXME what if length is < scan_axis
            return tuple(tmp)

        def transform_output(output_variable):
            # FIXME
            return output_variable

        # FIXME mutate current_variables in network to have new updates

        # create final output with the appropriate result
        network.create_vw(
            name="default",
            variable=transform_output(result_map[element_output.variable]),
            shape=transform_shape(element_output.shape),
        )
