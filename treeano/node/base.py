from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import six
import inspect
import abc
import types
import theano


from ..graph import TreeanoGraph
from ..update_deltas import UpdateDeltas
from ..variable import VariableWrapper


class MissingHyperparameter(Exception):
    pass


# ####################### handling serialization state #######################


CHILDREN_CONTAINERS = {}
NODES = {}


def register_node(name):
    """
    registers the decorated node with the given string for serialization
    """
    assert isinstance(name, six.string_types)

    def inner(cls):
        NODES[name] = cls
        return cls
    return inner


def node_to_str(cls):
    """
    returns the string for the given registered node class
    """
    return {v: k for k, v in NODES.iteritems()}[cls]


def node_from_str(s):
    """
    returns the registered node class for the given string
    """
    return NODES[s]


def node_to_data(node):
    """
    returns the given node as data
    """
    return dict(
        node_key=node_to_str(node.__class__),
        architecture_data=node._to_architecture_data(),
    )


def node_from_data(data):
    """
    convert the given node-representation-as-data back into an instance of
    the appropriate node class
    """
    node_key = data["node_key"]
    architecture_data = data["architecture_data"]
    return node_from_str(node_key)._from_architecture_data(architecture_data)


def register_children_container(name):
    """
    registers the decorated children container with the given string for
    serialization
    """
    assert isinstance(name, six.string_types)

    def inner(cls):
        CHILDREN_CONTAINERS[name] = cls
        return cls
    return inner


def children_container_to_str(cls):
    """
    returns the string for the given registered children container class
    """
    return {v: k for k, v in CHILDREN_CONTAINERS.iteritems()}[cls]


def children_container_from_str(s):
    """
    returns the registered children container class for the given string
    """
    return CHILDREN_CONTAINERS[s]


def children_container_to_data(cc):
    """
    returns the given children container as data
    """
    return dict(
        children_container_key=children_container_to_str(cc.__class__),
        children_container_data=cc.to_data(),
    )


def children_container_from_data(data):
    """
    convert the given children-container-representation-as-data back into an
    instance of the appropriate children-container class
    """
    cc_key = data["children_container_key"]
    cc_data = data["children_container_data"]
    return children_container_from_str(cc_key).from_data(cc_data)


# ############################ children container ############################


class ChildrenContainer(six.with_metaclass(abc.ABCMeta, object)):

    """
    API for dealing with the children of nodes (which are also nodes)
    """

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def to_data(self):
        """
        returns representation of container as data
        """

    @abc.abstractmethod
    def from_data(cls, data):
        """
        converts data representation back into an instance of the appropriate
        class

        NOTE: should be a classmethod
        """


@register_children_container("list")
class ListChildrenContainer(ChildrenContainer):

    """
    contains children as a list
    """

    def __init__(self, children):
        assert isinstance(children, (list, tuple))
        self.children = children

    def __iter__(self):
        return (x for x in self.children)

    def to_data(self):
        return [node_to_data(child_node) for child_node in self.children]

    @classmethod
    def from_data(cls, data):
        return cls([node_from_data(datum) for datum in data])


@register_children_container("none")
class NoneChildrenContainer(ChildrenContainer):

    """
    contains children as a list
    """

    def __init__(self, children):
        assert children is None

    def __iter__(self):
        return iter([])

    def to_data(self):
        return None

    @classmethod
    def from_data(cls, data):
        return cls(None)


@register_children_container("single_child")
class ChildContainer(ChildrenContainer):

    """
    contains a single child
    """

    def __init__(self, children):
        self.child = children

    def __iter__(self):
        return iter([self.child])

    def to_data(self):
        return node_to_data(self.child)

    @classmethod
    def from_data(cls, data):
        return cls(node_from_data(data))


# ######################### FIXME move to network.py #########################


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
            converts node names into their corresponding variables, with
            optional keys of which of the node's outputs to use
            """
            if isinstance(item, types.StringTypes):
                node_name = item
                return self.graph.output_for_node(node_name).variable
            elif isinstance(item, tuple):
                node_name, from_key = item
                return self.graph.output_for_node(node_name, from_key).variable
            else:
                # this should be a theano variable, because it is passed
                # into theano.function
                return item

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

# ########################### FIXME leave as base ###########################


class NodeAPI(six.with_metaclass(abc.ABCMeta, object)):

    """
    raw API for Node's in the graph
    """

    def __hash__(self):
        return hash((self.__class__, self.name))

    @abc.abstractproperty
    def name(self):
        """
        returns the node's name

        NOTE: should not change over time
        """

    @abc.abstractmethod
    def _to_architecture_data(self):
        """
        returns representation of the architecture data contained in the node
        """

    @abc.abstractmethod
    def _from_architecture_data(cls):
        """
        convert architecture data contained in a node into an instance
        of the appropriate class

        overriding this will allow for reading in old architectures in a
        backwards compatible manner

        NOTE: should be a classmethod
        """

    @abc.abstractmethod
    def get_hyperparameter(self, network, name):
        """
        returns the value of the given hyperparameter, if it is defined for
        the current node, otherwise raises a MissingHyperparameter
        """

    @abc.abstractmethod
    def architecture_children(self):
        """
        returns all child nodes of the given node in the architectural tree
        """

    @abc.abstractmethod
    def init_state(self, network):
        """
        defines all additional state (eg. parameters), possibly in a lazy
        manner, as well as additional dependencies

        also for defining other stateful values (eg. initialization schemes,
        update schemes)

        can assume that all dependencies have completed their init_state call

        NOTE: will be called exactly once
        """

    @abc.abstractmethod
    def get_input_keys(self, network):
        """
        returns the keys of the inputs to compute_output

        NOTE: should not change over time
        """

    @abc.abstractmethod
    def compute_output(self, network, *args):
        """
        computes output of a node as a dictionary from string key to
        output VariableWrapper

        NOTE: will be called exactly once
        """

    @abc.abstractmethod
    def mutate_update_deltas(self, network, update_deltas):
        """
        computes updates of a node and modifies the update_deltas that
        the network passed in

        NOTE: will be called exactly once, and should not mutate the input
        update_deltas, not return a new one
        """

    # ------------------------------------------------------------
    # utilities that are not part of the core API but nice to have
    # ------------------------------------------------------------

    def build(self):
        return build_network(self)


class NodeImpl(NodeAPI):

    """
    a nicer interface on top of the minimal NodeAPI, which sane defaults
    that could be overriden

    usage:
    - you probably do not want to override __init__

    example:
    class TreeaNode(NodeImpl):
        # have hyperparameters named a and b
        # default = no hyperparameters
        hyperparameter_names = ("a", "b")
        # specify a ChildrenContainer (if the node will have children)
        # default = no children
        children_container = ListChildrenContainer
        # define input keys to compute_output
        # default = ("default")
        input_keys = ("default", "something_else")

        def compute_output(self, network, in1, in2):
            # in1 and in2 correspond to input_keys
            assert in1.shape == in2.shape
            network.create_variable(
                name="default",
                variable=in1.variable + in2.variable,
                shape=in1.shape,
                tags={"output"},
            )
    node = TreeaNode("foobar", a=3, b=2)
    """

    # by default, have no hyperparameters
    hyperparameter_names = ()
    # by default, have no children
    children_container = NoneChildrenContainer
    # by default, have a single input_key of "default"
    input_keys = ("default",)

    def __init__(self, name, children=None, **kwargs):
        self._name = name
        self.children = self.children_container(children)
        self.hyperparameters = kwargs
        # some validation
        assert isinstance(self.children, ChildrenContainer)
        assert isinstance(self.input_keys, (list, tuple))
        assert isinstance(self.hyperparameter_names, (list, tuple, set))
        for key in kwargs:
            assert key in self.hyperparameter_names, dict(
                name=self.name,
                key=key,
                msg="Incorrect hyperparameter"
            )

    def __repr__(self):
        param_pairs = [("name", self.name)]
        param_pairs.extend(self.hyperparameters.iteritems())
        param_str = ", ".join("%s=%s" % (k, repr(v)) for k, v in param_pairs)
        root = "%s(%s)" % (self.__class__.__name__, param_str)
        # OPTIMIZE
        children = [repr(child).replace("\n", "\n| ")
                    for child in self.children]
        children_str = "\n| ".join([root] + children)
        return children_str

    @property
    def name(self):
        return self._name

    def _to_architecture_data(self):
        return dict(
            name=self.name,
            children=children_container_to_data(self.children),
            hyperparameters=self.hyperparameters,
        )

    @classmethod
    def _from_architecture_data(cls, data):
        return cls(
            name=data['name'],
            children=children_container_from_data(data["children"]),
            **data['hyperparameters']
        )

    def get_hyperparameter(self, network, name):
        """
        default implementation that uses the values in self.hyperparameters
        """
        if name in self.hyperparameters:
            return self.hyperparameters[name]
        else:
            raise MissingHyperparameter

    def architecture_children(self):
        """
        by default, return children in children_container
        """
        return list(iter(self.children))

    def init_state(self, network):
        """
        by default, do nothing with the state
        """

    def get_input_keys(self, network):
        """
        by default, use values in self.input_keys
        """
        return self.input_keys

    def compute_output(self, network, *args):
        """
        by default, return first input as output
        """
        network.copy_variable(
            name="default",
            previous_variable=args[0],
            tags={"output"},
        )

    def new_update_deltas(self, network):
        """
        an alternative API for providing update deltas without mutation
        """
        return UpdateDeltas()

    def mutate_update_deltas(self, network, update_deltas):
        """
        default implementation of mutate_update_deltas that uses
        new_update_deltas
        """
        update_deltas += self.new_update_deltas(network)


class WrapperNodeImpl(NodeImpl):

    """
    a nicer interface on top of the minimal NodeAPI, which sane defaults
    that could be overriden - specifically for nodes that wrapper other nodes

    usage:
    - you probably do not want to override __init__
    """
    # by default, children is a list
    children_container = ListChildrenContainer
    # by default, input keys return default input and the value created
    # when calling take_output_from
    input_keys = ("final_child_output",)

    def init_state(self, network):
        """
        by default, forward input to first child, and take output from last
        child
        """
        children = self.architecture_children()
        network.forward_input_to(children[0].name)
        network.take_output_from(children[-1].name,
                                 to_key="final_child_output")

    def compute_output(self, network, child_output):
        """
        by default, returns the value from the final_child
        """
        network.copy_variable(
            name="default",
            previous_variable=child_output,
            tags={"output"},
        )


class Wrapper1NodeImpl(WrapperNodeImpl):

    """
    a nicer interface on top of the minimal NodeAPI, which sane defaults
    that could be overriden - specifically for nodes that wrap a single node

    usage:
    - you probably do not want to override __init__
    """
    # by default, children is a single node
    children_container = ChildContainer

# ############################# FIXME DEPRECATED #############################


class Node(object):

    # """
    # all nodes require a unique name attribute as a string
    # """

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
        for node in graph.architectural_tree_nodes_root_to_leaves():
            node.variables = []
        # initialize state
        # ---
        # this is reversed so that outer nodes have their state initialized
        # before inner nodes - this is important for sequential nodes, since
        # the first child will depend on the input of the sequential node, and
        # we would like to make that dependency explicit
        for node in graph.architectural_tree_nodes_root_to_leaves():
            # share graph between all nodes
            node.graph = graph
            # recursively init nodes
            node.init_state()
        # compute and cache outputs
        for node in graph.computation_graph_nodes_topological():
            output = node.compute_output()
            node.output = output
        # sets that updates have not yet been computed
        for node in graph.architectural_tree_nodes_leaves_to_root():
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
        variable = VariableWrapper(new_name,
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
            for node in self.graph.architectural_tree_nodes_root_to_leaves():
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
        manner, as well as additional dependencies

        also for defining other stateful values (eg. initialization schemes,
        update schemes)

        optional to define (no need to define if the node is stateless)

        can assume that all dependencies have completed their init_state call
        """

    def compute_output(self):
        """
        computes output of a node as a dictionary from string key to
        output VariableWrapper
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
