import types

import theano

from .graph import TreeanoGraph
from .update_deltas import UpdateDeltas
from .variable import VariableWrapper


class MissingHyperparameter(Exception):
    pass


class Network(object):

    """
    contains the state of multiple nodes
    """

    def __init__(self, root_node):
        self.root_node = root_node
        self.node_state = {}
        self.update_deltas = UpdateDeltas()

    def build(self):
        """
        initialize network state
        """
        self.graph = TreeanoGraph(self.root_node)
        # set node state for each node to be empty
        # ---
        # order doesn't matter
        for node in reversed(self.graph.nodes(order="architecture")):
            node_state = {}
            # initialize some parts of node_state
            node_state["current_variables"] = {}
            node_state["original_variables"] = {}
            self.node_state[node.name] = node_state
        # initialize state
        # ---
        # outer nodes have their state initialized
        # before inner nodes - this is important for sequential nodes, since
        # the first child will depend on the input of the sequential node, and
        # we would like to make that dependency explicit
        for node in self.graph.architectural_tree_nodes_root_to_leaves():
            node.init_state(self.relative_network(node))
        # freeze computation graph
        # ---
        # if a node changes the computation graph while traversing it,
        # there is a chance that the relevant nodes have already been processed
        # thus being a likely source of error
        self.graph.is_mutable = False
        # compute and store outputs
        # ---
        # compute in the order of the computation DAG, so that all
        # dependencies have been computed for each node by the time
        # computation for the node has to occur
        for node in self.graph.computation_graph_nodes_topological():
            rel_network = self.relative_network(node)
            # get input keys
            input_keys = node.get_input_keys(rel_network)
            # lookup input variables
            inputs = []
            for input_key in input_keys:
                # find which node our input comes from, and the name of
                # the variable containing the input
                node_name, from_key = self.graph.input_edge_for_node(node.name,
                                                                     input_key)
                inputs.append(self[node_name].get_variable(from_key))
            # store input variables for the node
            # ---
            # there is no immediate reason to do so, but doing it just in case
            # for now
            rel_network.store_inputs(dict(zip(input_keys, inputs)))
            # compute outputs
            output_res = node.compute_output(rel_network, *inputs)
            # sanity check to make sure no user accidentaly returns a value
            # instead of creating a variable
            assert output_res is None
        # compute updates
        # ---
        # compute from top (root) to bottom (leaves) so that low levels
        # of the tree (ie. more specific update rules) can overwrite / mutate
        # the update rules from higher leveles of the tree (ie. more general
        # update rules)
        for node in self.graph.architectural_tree_nodes_root_to_leaves():
            node.mutate_update_deltas(self.relative_network(node),
                                      self.update_deltas)

    def relative_network(self, node):
        """
        returns a network relative to a single node
        """
        return RelativeNetwork(self, node)

    def __getitem__(self, node_name):
        """
        sugar for accessing nodes in a graph
        """
        node = self.graph.name_to_node[node_name]
        return self.relative_network(node)

    def function(self,
                 inputs,
                 outputs=None,
                 include_updates=False,
                 updates=None,
                 **kwargs):
        """
        wrapper around theano.function that allows reference node outputs
        with strings

        example:
        network.function(["input_node"], ["fc_node", "loss", ("conv1", "W")])
        """
        if outputs is None:
            outputs = []
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)

        if include_updates:
            # combine update_deltas with manually specified updates
            if updates is None:
                all_deltas = self.update_deltas
            else:
                extra_updates = UpdateDeltas.from_updates(updates)
                all_deltas = self.update_deltas + extra_updates

            # convert into format expected by theano.function
            updates = all_deltas.to_updates()

        def transform(item):
            """
            converts node names into their corresponding theano variables,
            with optional keys of which of the node's outputs to use
            """
            if isinstance(item, types.StringTypes):
                node_name = item
                from_key = "default"
            elif isinstance(item, tuple):
                node_name, from_key = item
            else:
                # this should be a theano variable, because it is passed
                # into theano.function
                return item

            return self[node_name].get_variable(from_key).variable

        transformed_inputs = map(transform, inputs)
        transformed_outputs = map(transform, outputs)

        fn = theano.function(inputs=transformed_inputs,
                             outputs=transformed_outputs,
                             updates=updates,
                             **kwargs)
        return fn


class NoDefaultValue(object):
    pass


class RelativeNetwork(object):

    """
    network relative to a single node
    """

    def __init__(self, network, node):
        self._network = network
        self._node = node
        self._name = node.name
        self._state = self.network.node_state[self._name]

    def __getattr__(self, name):
        """
        by default, behave like the non-relative network
        """
        return getattr(self._network, name)

    def store_inputs(self, inputs):
        self._state["inputs"] = inputs

    def get_variable(self, variable_name):
        return self._state["current_variables"][variable_name]

    def find_hyperparameter(self,
                            hyperparameter_keys,
                            default_value=NoDefaultValue):
        """
        throws an exception if no default value is given

        example:
        >>> network.find_hyperparameter(["foo", "bar", "choo"], 42)

        the network first search all ancestors for a hyperparamter named
        "foo". if that isn't found, it searches for a hyperparameter named
        "bar", and if that isn't found returns 42
        """
        ancestors = list(self.graph.architecture_ancestors(self._name))
        for hyperparameter_key in hyperparameter_keys:
            for node in [self._node] + ancestors:
                try:
                    value = node.get_hyperparameter(hyperparameter_key)
                except MissingHyperparameter:
                    pass
                else:
                    return value
        if default_value is NoDefaultValue:
            raise MissingHyperparameter(dict(
                hyperparameter_keys=hyperparameter_keys,
            ))
        else:
            return default_value

    def find_variables_in_subtree(self, tag_filters):
        """
        return variables matching all of the given tags
        """
        tag_filters = set(tag_filters)
        return [variable
                for node in self.graph.architecture_subtree(self._name)
                for variable in node.variables
                # only keep variables where all filters match
                if len(tag_filters - variable.tags) == 0]

    def create_variable(self, name, **kwargs):
        """
        creates a new output variable for the current node
        """
        # we don't want to overwrite an existing value
        assert name not in self._state['current_variables']
        assert name not in self._state['original_variables']
        new_name = "%s.%s" % (self._name, name)
        # prepare initialization strategies
        inits = self.find_hyperparameter(["shared_initializations"],
                                         default_value=[])
        kwargs["shared_initializations"] = inits
        kwargs["relative_network"] = self
        # create the variable
        variable = VariableWrapper(new_name, **kwargs)
        # save variable
        self._state['current_variables'][name] = variable
        self._state['original_variables'][name] = variable
        return variable

    def copy_variable(self, name, previous_variable, tags=None):
        """
        creates a copy of previous_variable under a new name

        the main use case for this is for wrapper nodes which just pass
        their input as their output
        """
        return self.create_variable(
            name,
            variable=previous_variable.variable,
            shape=previous_variable.shape,
            tags=tags,
        )

    def replace_variable(self, name, new_variable):
        """
        replaces the given variable for a node in 'current_variables' state
        with a new variable

        NOTE: this is design for use with scan, so that non-sequence variables
        can be replaced by their sequence versions
        """
        assert name in self._state['original_variables']
        self._state['current_variables'][name] = new_variable
        return new_variable

    def forward_input_to(self,
                         node_name,
                         previous_to_key="default",
                         to_key="default",
                         ignore_no_input=True):
        """
        forwards input of current node, if any, to a new node with the
        given node_name (presumable a child_node)

        the main use case for this would be to have the input of a container
        be sent to one of its children
        """
        input_edge = self.graph.input_edge_for_node(self._name,
                                                    to_key=previous_to_key)
        # there may not be an input
        # (eg. if the wrapper node is holding the input node)
        if input_edge is None:
            if not ignore_no_input:
                raise ValueError("forward_input_to called on node without "
                                 "input key: %s" % previous_to_key)
            else:
                # ignore the issue and do nothing
                pass
        else:
            name_from, from_key = input_edge
            self.graph.add_dependency(name_from,
                                      node_name,
                                      from_key=from_key,
                                      to_key=to_key)

    def take_output_from(self,
                         node_name,
                         from_key="default",
                         to_key="default"):
        """
        forwards output of a given node (with key from_key) to the current
        node (with key to_key)

        the main use case for this would be to have the output of a child of
        a container node be sent to the container to allow it to propagate
        forward in the DAG
        """
        self.graph.add_dependency(node_name,
                                  self._name,
                                  from_key=from_key,
                                  to_key=to_key)


def build_network(root_node):
    network = Network(root_node)
    network.build()
    return network
