from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import inspect
import types
import theano


from ..graph import TreeanoGraph
from ..update_deltas import UpdateDeltas
from ..variable import LazyWrappedVariable


class Node(object):

    """
    all nodes require a unique name attribute as a string
    """

    def __hash__(self):
        return hash((self.__class__, self.name))

    @classmethod
    def from_architecture_data(cls, data):
        """
        convert architecture data contained in a node into an instance
        of the appropriate class

        by default, simply call the constructor

        overriding this will allow for reading in old architectures in a
        backwards compatible manner
        """
        # FIXME see comment in to_architecture_data
        return cls(**data)

    # TODO maybe make a separate function
    @classmethod
    def constructor_arguments(cls):
        """
        returns constructor arguments of the class
        """
        argspec = inspect.getargspec(cls.__init__)
        args = argspec.args
        assert args[0] == "self"
        return args[1:]

    def to_architecture_data(self):
        """
        returns representation of the architecture data contained in the node

        by default, this is the state from the constructor arguments

        generally shouldn't be overridden, but can be if one wants to contol
        the serialization structure
        """
        # FIXME this only works for leaf nodes - need to convert composite
        # nodes into data as well
        return {arg: self.__dict__[arg]
                for arg in self.constructor_arguments()}

    def architecture_copy(self):
        """
        returns a shallow copy of the architecture of the node
        """
        # FIXME HACK to get around not implementing this properly yet
        import copy
        return copy.deepcopy(self)
        return self.from_architecture_data(self.to_architecture_data())

    def build(self):
        """
        builds a mutable copy of the network with state initialized
        """
        # make copy of nodes
        root_node = self.architecture_copy()
        # build computation graph
        graph = TreeanoGraph(root_node)
        # add attribute for variables
        for node in reversed(graph.nodes(order="architecture")):
            node.variables = []
        # create a shared mapping from theano variable to lazy wrapped variable
        # ---
        # this stores the state of all the lazy wrapped variables in the
        # network
        variable_map = {}
        for node in reversed(graph.nodes(order="architecture")):
            node.variable_map = variable_map
        # initialize state
        # ---
        # this is reversed so that outer nodes have their state initialized
        # before inner nodes - this is important for sequential nodes, since
        # the first child will depend on the input of the sequential node, and
        # we would like to make that dependency explicit
        for node in reversed(graph.nodes(order="architecture")):
            # share graph between all nodes
            node.graph = graph
            # recursively init nodes
            node.init_state()
        # compute and cache outputs
        for node in graph.nodes(order="computation"):
            output = node.compute_output()
            node.output = output
        # sets that updates have not yet been computed
        for node in graph.nodes(order="architecture"):
            node.updates_computed = False
        return root_node

    def __getitem__(self, node_name):
        """
        sugar for accessing nodes in a graph
        """
        return self.graph.name_to_node[node_name]

    def get_input(self, to_key="default"):
        """
        returns input of the current node in the graph for a given key
        """
        return self.graph.input_for_node(self.name, to_key)

    def create_variable(self, name, *args, **kwargs):
        # we don't want to overwrite an existing attribute
        assert not hasattr(self, name)
        new_name = "%s.%s" % (self.name, name)
        # prepare initialization strategies
        inits = self.find_hyperparameter("shared_initializations", [])
        kwargs["shared_initializations"] = inits
        # create the variable
        variable = LazyWrappedVariable(new_name,
                                       self.variable_map,
                                       *args,
                                       **kwargs)
        # set variable as attribute for easy access
        setattr(self, name, variable)
        # register variable for future searching of parameters
        self.variables.append(variable)

    def compute_all_update_deltas(self):
        """
        computes and caches the update deltas of each node

        this is not done in Node.build because the computation done may
        be unnecessary
        """
        # if updates are already computed for this node, we are done
        if not self.updates_computed:
            update_deltas = UpdateDeltas()
            # perform this tree walk top-down (from root to leaves), so that
            # lower (more-specific) nodes can manipulate the updates of higher
            # (more-general) nodes
            for node in reversed(self.graph.nodes(order="architecture")):
                # no node should have updates computed already if any node
                # hasn't had it's updates computed
                assert not node.updates_computed
                node.updates_computed = True
                node.update_deltas = update_deltas
                res = node.compute_update_deltas(update_deltas)
                # we want to make sure there is no confusion that the API
                # should return a new value instead of mutating the passed in
                # value
                assert res is None
        return self.update_deltas

    def function(self,
                 inputs,
                 outputs=None,
                 generate_updates=False,
                 updates=None,
                 **kwargs):
        if outputs is None:
            outputs = []
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)

        if generate_updates:
            # compute and cache updates
            all_deltas = self.compute_all_update_deltas()

            # combine with manually specified updates
            if updates is not None:
                update_deltas = UpdateDeltas.from_updates(updates)
                all_deltas = all_deltas + update_deltas

            # convert into format expected by theano.function
            updates = all_deltas.to_updates()

        def transform(item):
            """
            converts node names into their corresponding variables, with
            optional keys of which of the node's outputs to use
            """
            if isinstance(item, types.StringTypes):
                return self.graph.output_for_node(item).variable
            elif isinstance(item, tuple):
                return self.graph.output_for_node(*item).variable
            else:
                return item

        transformed_inputs = map(transform, inputs)
        transformed_outputs = map(transform, outputs)

        fn = theano.function(inputs=transformed_inputs,
                             outputs=transformed_outputs,
                             updates=updates,
                             **kwargs)
        return fn

    def find_variables_in_subtree(self, tag_filters):
        """
        return variables matching all of the given tags
        """
        tag_filters = set(tag_filters)
        return [variable
                for node in self.graph.architecture_subtree(self.name)
                for variable in node.variables
                # only keep variables where all filters match
                if len(tag_filters - variable.tags) == 0]

    def find_hyperparameter(self, hyperparameter_name, default=None):
        """
        recursively search up the architectural tree for a node with the
        given hyperparameter_name specified (returns the default if not
        provided)
        """
        nodes = [self] + list(self.graph.architecture_ancestors(self.name))
        for node in nodes:
            value = node.get_hyperparameter(hyperparameter_name)
            if value is not None:
                return value
        if default is not None:
            return default
        raise ValueError("Hyperparameter value of %s not specified in the "
                         "current architecture"
                         % hyperparameter_name)

    def get_hyperparameter(self, hyperparameter_name):
        """
        returns the value of the given hyperparameter, if it is defined for
        the current node, otherwise returns None

        optional to define (no need to define if node has no hyperparameters)
        """
        if hyperparameter_name in self.constructor_arguments():
            return getattr(self, hyperparameter_name)
        else:
            return None

    def architecture_children(self):
        """
        returns all child nodes of the given node

        optional to define (no need to define if node has no children)
        """
        # TODO maybe allow for a filter here by node type
        return []

    def children_names(self):
        """
        returns names of all child nodes of the given node
        """
        return [child.name for child in self.architecture_children()]

    def init_state(self):
        """
        defines all additional state (eg. parameters), possibly in a lazy
        manner

        also for defining other stateful values (eg. initialization schemes,
        update schemes)

        optional to define (no need to define if the node is stateless)

        can assume that all dependencies have completed their init_state call
        """

    def compute_output(self):
        """
        computes output of a node as a dictionary from string key to
        output LazyWrappedVariable
        """
        raise NotImplementedError

    def compute_update_deltas(self, update_deltas):
        """
        computes updates of a node and modifies the update_deltas that
        the network passed in

        optional to define - if the node doesn't update itself
        """


class WrapperNode(Node):

    """
    mixin to provide useful methods for nodes that wrap other nodes
    """

    def architecture_children(self):
        raise NotImplementedError

    def forward_input_to(self, child_name, to_key="default"):
        """
        forwards input of current node, if any, to the child node with the
        given name and to_key
        """
        input_edge = self.graph.input_edge_for_node(self.name)
        # there may not be an input
        # (eg. if the wrapper node is holding the input node)
        self.wrapper_has_input = input_edge is not None
        if self.wrapper_has_input:
            name_from, from_key = input_edge
            self.graph.add_dependency(name_from,
                                      child_name,
                                      from_key=from_key,
                                      to_key=to_key)

    def take_input_from(self, child_name, from_key="default"):
        self.graph.add_dependency(child_name,
                                  self.name,
                                  to_key="wrapper_output")

    def compute_output(self):
        """
        by default, returns the value created when calling take_input_from
        """
        return dict(
            default=self.get_input(to_key="wrapper_output")
        )
