import six
import theano
import theano.tensor as T

from .graph import TreeanoGraph
from .update_deltas import UpdateDeltas
from .variable import VariableWrapper
from .. import utils


class MissingHyperparameter(Exception):
    pass


class Network(object):

    """
    contains the state of multiple nodes
    """

    def __init__(self,
                 root_node,
                 override_hyperparameters=None,
                 default_hyperparameters=None):
        self.root_node = root_node
        self.node_state = {}
        self.update_deltas = UpdateDeltas()
        self.override_hyperparameters = dict()
        self.default_hyperparameters = dict(
            batch_axis=0,
            deterministic=False,
            monitor=True,
        )
        if override_hyperparameters is not None:
            self.override_hyperparameters.update(override_hyperparameters)
        if default_hyperparameters is not None:
            self.default_hyperparameters.update(default_hyperparameters)

    @property
    def is_built(self):
        return hasattr(self, "graph")

    def build(self):
        """
        initialize network state
        """
        # make building idempotent
        # ---
        # this allows building to be lazy. this way, if we don't have to
        # do all the work until it is needed
        # example use case: sequentially applying several transforms to a
        # network
        if self.is_built:
            return
        self.graph = TreeanoGraph(self.root_node)
        # set node state for each node to be empty
        # ---
        # order doesn't matter
        for node in self.graph.architectural_tree_nodes_root_to_leaves():
            node_state = {}
            # initialize some parts of node_state
            node_state["current_variables"] = {}
            node_state["original_variables"] = {}
            node_state["additional_data"] = {}
            node_state["set_hyperparameters"] = {}
            self.node_state[node.name] = node_state
        # initialize long range dependencies
        # ---
        # order doesn't matter
        # done before init_state because some nodes need to know their inputs
        for node in self.graph.architectural_tree_nodes_root_to_leaves():
            node.init_long_range_dependencies(self.relative_network(node))
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
                inputs.append(rel_network.get_input_vw(input_key))
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

    def relative_network(self, node=None):
        """
        returns a network relative to a single node
        """
        self.build()
        if node is None:
            node = self.root_node
        return RelativeNetwork(self, node)

    def __contains__(self, node_name):
        """
        sugar for checking if a node name is in the graph
        """
        self.build()
        return node_name in self.graph.name_to_node

    def __getitem__(self, node_name):
        """
        sugar for accessing nodes in a graph
        """
        self.build()
        node = self.graph.name_to_node[node_name]
        return self.relative_network(node)

    def network_variable(self, query):
        """
        converts node names into their corresponding theano variables,
        with optional keys of which of the node's outputs to use

        eg.
        network.network_variable("input")
        network.network_variable(("fc1", "W"))
        network.network_variable(var)  # no-op
        """
        if isinstance(query, six.string_types):
            node_name = query
            from_key = "default"
        elif isinstance(query, tuple):
            node_name, from_key = query
        else:
            # this should be a theano variable
            return query

        return self[node_name].get_vw(from_key).variable

    def function(self,
                 inputs,
                 outputs=None,
                 include_updates=False,
                 updates=None,
                 givens=None,
                 **kwargs):
        """
        wrapper around theano.function that allows reference node outputs
        with strings

        example:
        network.function(["input_node"], ["fc_node", "loss", ("conv1", "W")])
        """
        self.build()
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

        transformed_inputs = [self.network_variable(i) for i in inputs]
        transformed_outputs = [self.network_variable(i) for i in outputs]

        if givens is None:
            tmp_givens = []
        elif isinstance(givens, dict):
            tmp_givens = list(givens.items())
        elif isinstance(givens, (list, tuple)):
            tmp_givens = list(givens)
        transformed_givens = [(self.network_variable(k), v)
                              for k, v in tmp_givens]
        fn = theano.function(inputs=transformed_inputs,
                             outputs=transformed_outputs,
                             updates=updates,
                             givens=transformed_givens,
                             **kwargs)
        return fn

    def is_relative(self):
        return False


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
        self._state = self._network.node_state[self._name]

    def __getattr__(self, name):
        """
        by default, behave like the non-relative network
        """
        return getattr(self._network, name)

    def __getitem__(self, name):
        return self._network[name]

    def is_relative(self):
        return True

    def store_inputs(self, inputs):
        """
        stores the inputs for the current node
        """
        self._state["inputs"] = inputs

    def set_data(self, key, value):
        # we don't want ambiguity with names, thus don't allow
        # the same name as a variable, and also don't allow overwriting
        # additional_data
        assert key not in self._state["additional_data"]
        assert key not in self._state["current_variables"]
        self._state["additional_data"][key] = value

    def get_data(self, key):
        return self._state["additional_data"][key]

    def get_vw(self, variable_name):
        return self._state["current_variables"][variable_name]

    def set_hyperparameter(self, node_name, key, value):
        """
        sets a hyperparameter for a child node
        """
        if node_name not in self._state["set_hyperparameters"]:
            self._state["set_hyperparameters"][node_name] = {}
        self._state["set_hyperparameters"][node_name][key] = value

    def forward_hyperparameter(self,
                               node_name,
                               key,
                               hyperparameter_keys,
                               *args,
                               **kwargs):
        """
        forwards a set of hyperparameters to a different node under a different
        key
        """
        value = self.find_hyperparameter(hyperparameter_keys, *args, **kwargs)
        self.set_hyperparameter(node_name, key, value)

    def maybe_forward_hyperparameter(self, *args, **kwargs):
        """
        forwards hyperaparameters to a different node, if given

        returns True if set, False otherwise
        """
        try:
            self.forward_hyperparameter(*args, **kwargs)
            return True
        except MissingHyperparameter:
            return False

    def find_hyperparameter(self,
                            hyperparameter_keys,
                            default_value=NoDefaultValue):
        """
        throws an exception if no default value is given

        example:
        >>> network.find_hyperparameter(["foo", "bar", "choo"], 42)

        the network first looks at override_hyperparameters, then searches the
        current node for hyperparameters named "foo", "bar", or "choo" in that
        order, then looks at the ancestor of the current node, repeating until
        out of nodes. if no ancestor has a hyperparameter for one of the keys
        42 is returned
        """
        # return first valid hyperparameter
        for val in self.find_hyperparameters(hyperparameter_keys,
                                             default_value):
            return val
        else:
            # otherwise, raise an exception
            raise MissingHyperparameter(dict(
                hyperparameter_keys=hyperparameter_keys,
            ))

    def find_hyperparameters(self,
                             hyperparameter_keys,
                             default_value=NoDefaultValue):
        """
        returns generator of all hyperparameters for the given keys
        in the order of precedence
        """
        # use override_hyperparameters
        # ---
        # this has highest precedence
        for hyperparameter_key in hyperparameter_keys:
            if hyperparameter_key in self.override_hyperparameters:
                yield self.override_hyperparameters[hyperparameter_key]
        # look through hyperparameters of all ancestors
        ancestors = list(self.graph.architecture_ancestors(self._name))
        # prefer closer nodes over more specific queries
        done_ancestors_names = []
        for node in [self._node] + ancestors:
            # append current node to done ancestor
            # ---
            # this is done before the loop, so a node can set_hyperparameter
            # for itself
            done_ancestors_names.append(node.name)
            # prepare set_hyperparameters state
            node_hps = self.node_state[node.name]["set_hyperparameters"]
            for hyperparameter_key in hyperparameter_keys:
                # try finding set hyperparameters
                for ancestor_name in done_ancestors_names:
                    try:
                        yield node_hps[ancestor_name][hyperparameter_key]
                    except KeyError:
                        pass
                # try finding provided hyperparameters
                try:
                    yield node.get_hyperparameter(self, hyperparameter_key)
                except MissingHyperparameter:
                    pass
        # try returning the given default value, if any
        if default_value is not NoDefaultValue:
            yield default_value
        # try global default hyperparameters
        # ---
        # this has lowest precedence
        for hyperparameter_key in hyperparameter_keys:
            if hyperparameter_key in self.default_hyperparameters:
                yield self.default_hyperparameters[hyperparameter_key]

    def find_vws_in_subtree(self, tags=None, is_shared=None):
        """
        return variable wrappers matching all of the given tags
        """
        remaining_vws = [
            vw
            for name in self.graph.architecture_subtree_names(self._name)
            for vw in self[name]._state["current_variables"].values()]
        if tags is not None:
            tags = set(tags)
            # only keep variables where all tags match
            remaining_vws = [vw for vw in remaining_vws
                             if len(tags - vw.tags) == 0]
        if is_shared is not None:
            remaining_vws = [vw for vw in remaining_vws
                             if is_shared == vw.is_shared]
        return remaining_vws

    def find_nodes_in_subtree(self, cls):
        """
        return all nodes with the given class
        """
        return [node for node in self.graph.architecture_subtree(self._name)
                if node.__class__ is cls]

    def create_vw(self, name, **kwargs):
        """
        creates a new output variable for the current node
        """
        # we don't want to overwrite an existing value
        assert name not in self._state['current_variables'], name
        assert name not in self._state['original_variables'], name
        # FIXME have a defined name separator
        new_name = "%s:%s" % (self._name, name)
        # same metadata about the network
        kwargs["relative_network"] = self
        # create the variable
        variable = VariableWrapper(new_name, **kwargs)
        # save variable
        self._state['current_variables'][name] = variable
        self._state['original_variables'][name] = variable
        return variable

    def copy_vw(self, name, previous_vw, tags=None):
        """
        creates a copy of previous_vw under a new name

        the main use case for this is for wrapper nodes which just pass
        their input as their output
        """
        variable = previous_vw.variable
        if utils.is_shared_variable(variable):
            # creating a "copy"of the variable
            # ---
            # rationale: shared variables are seen as normal variables
            # (so shared variables aren't accidentaly owned by multiple nodes)
            # ---
            # why tensor_copy? theano.compile.view_op doesn't support Rop
            variable = T.tensor_copy(variable)
        return self.create_vw(
            name,
            variable=variable,
            shape=previous_vw.shape,
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
        self.add_dependency(node_name,
                            self._name,
                            from_key=from_key,
                            to_key=to_key)

    def forward_output_to(self,
                          node_name,
                          from_key="default",
                          to_key="default"):
        """
        forwards output of the current node (with key from_key) to the given
        node (with key to_key)
        """
        self.add_dependency(self._name,
                            node_name,
                            from_key=from_key,
                            to_key=to_key)

    def add_dependency(self,
                       from_name,
                       to_name,
                       from_key="default",
                       to_key="default"):
        """
        wrapper around self.graph.add_dependency
        """
        self.graph.add_dependency(from_name=from_name,
                                  to_name=to_name,
                                  from_key=from_key,
                                  to_key=to_key)

    def remove_dependency(self, from_name, to_name):
        """
        wrapper around self.graph.remove_dependency
        """
        self.graph.remove_dependency(from_name=from_name,
                                     to_name=to_name)

    def get_all_input_edges(self):
        """
        returns a map from input keys of the current node to the node where
        the edge is from
        """
        edges = {}
        for edge in self.graph.all_input_edges_for_node(self._name):
            edge_from, edge_to, datamap = edge
            assert edge_to == self._name
            to_key = datamap.get("to_key")
            if to_key is not None:
                edges[to_key] = edge_from
        return edges

    def get_input_vw(self, input_key):
        """
        return vw for given input key
        """
        node_name, from_key = self.graph.input_edge_for_node(self._name,
                                                             input_key)
        return self[node_name].get_vw(from_key)
