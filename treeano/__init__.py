from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import inspect
import types
import operator
import numpy as np
import theano
import theano.tensor as T
from fields import Fields
import toolz
import networkx as nx
import lasagne

floatX = theano.config.floatX
ENABLE_TEST_VALUE = theano.config.compute_test_value != "off"

# TODO move to update_delta module


class UpdateDeltas(object):

    def __init__(self, deltas):
        assert isinstance(deltas, dict)
        self.deltas = deltas

    def to_updates(self):
        updates = []
        for var, delta in self.deltas.items():
            updates.append((var, var + delta))
        # sorting updates by name so that the order is deterministic
        updates.sort(key=lambda pair: pair[0].name)
        return updates

    @classmethod
    def from_updates(cls, updates):
        if isinstance(updates, list):
            delta_dict = {var: (new_value - var)
                          for var, new_value in updates}
        elif isinstance(updates, dict):
            delta_dict = {var: (new_value - var)
                          for var, new_value in updates.items()}
        else:
            raise ValueError("Can't handle updates of the given type")
        return cls(delta_dict)

    def apply(self, fn):
        return UpdateDeltas({k: fn(v) for k, v in self.deltas.items()})

    def __add__(self, other):
        if isinstance(other, UpdateDeltas):
            return UpdateDeltas(toolz.merge_with(sum,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: x + other)

    def __mul__(self, other):
        if isinstance(other, UpdateDeltas):
            def product(iterable):
                return reduce(operator.mul, iterable, 1)
            # TODO this will currently make it such that if one instance
            # has updates and another doesn't, it will return the same value
            # (another approach would be returning 0 if the value isn't in
            # both)
            # TODO is multiply by another set of deltas ever desired?
            return UpdateDeltas(toolz.merge_with(product,
                                                 self.deltas,
                                                 other.deltas))
        else:
            return self.apply(lambda x: x * other)

x = theano.shared(0, name="x")
ud = UpdateDeltas({x: 0})
ud += 1
ud *= 2
fn = theano.function([], updates=ud.to_updates())
fn()
assert x.get_value() == 2
fn()
assert x.get_value() == 4


# TODO move to initialization module

# ############################### base classes ###############################


class SharedInitialization(object):

    """
    interface for initialization schemes of shared variables
    """

    def predicate(self, var):
        """
        whether or not the current initialization applies to the current
        variable
        """
        return True

    def create_shared(self, var):
        """
        creates the shared variable with an appropriately initialized value
        """
        kwargs = {}
        if len(var.broadcastable) > 0:
            kwargs["broadcastable"] = var.broadcastable
        value = self.initialize_value(var)
        kwargs["name"] = var.name
        kwargs["value"] = np.array(value).astype(var.dtype)
        variable = theano.shared(**kwargs)
        return variable

    def initialize_value(self, var):
        """
        creates appropriately initialized value for the given
        LazyWrappedVariable
        """
        raise NotImplementedError


class WeightInitialization(SharedInitialization):

    """
    base class for initializations that only work on weights
    """

    def predicate(self, var):
        return "weight" in var.tags

# ############################# implementations #############################


class ExceptionInitialization(SharedInitialization):

    """
    initialization scheme that always throws an exception - so that
    initialization doesn't fall back to other schemes
    """

    def initialize_value(self, var):
        assert False, "Initialization failed"


class ZeroInitialization(SharedInitialization):

    """
    initializes shared variable to zeros
    """

    def initialize_value(self, var):
        if len(var.broadcastable) > 0:
            value = np.zeros(var.shape)
        else:
            value = 0
        return value


class PreallocatedInitialization(SharedInitialization):

    """
    uses already defined shared variables and does NOT overwrite their
    values
    """

    def __init__(self, name_to_shared):
        self.name_to_shared = name_to_shared

    def predicate(self, var):
        return var.name in self.name_to_shared

    def create_shared(self, var):
        shared = self.name_to_shared[var.name]
        assert shared.dtype == var.dtype
        assert shared.get_value().shape == var.shape
        assert shared.name == var.name
        assert shared.broadcastable == var.broadcastable
        return shared


class GlorotUniform(WeightInitialization):

    # FIXME add parameters to constructor (eg. initialization_gain)

    def initialize_value(self, var):
        return lasagne.init.GlorotUniform().sample(var.shape).astype(var.dtype)


# TODO move to variable module

VALID_TAGS = set("""
input
weight
bias
parameter
monitor
state
""".split())


class LazyWrappedVariable(object):

    def __init__(self,
                 name,
                 variable_map=None,
                 shape=None,
                 dtype=None,
                 broadcastable=None,
                 is_shared=None,
                 tags=None,
                 ndim=None,
                 variable=None,
                 shared_initializations=None):
        self.name = name
        self.variable_map = variable_map
        self.shape_ = shape
        self.dtype_ = dtype
        self.broadcastable_ = broadcastable
        self.is_shared_ = is_shared
        self.tags_ = tags
        self.ndim_ = ndim
        self.variable_ = variable
        self.shared_initializations = shared_initializations
        self.validate()

    def validate(self):
        shape = self.shape_
        dtype = self.dtype_
        broadcastable = self.broadcastable_
        is_shared = self.is_shared_
        tags = self.tags_
        ndim = self.ndim_
        variable = self.variable_

        if ndim is not None and shape is not None:
            assert len(shape) == ndim
        if ndim is not None and variable is not None:
            assert ndim == variable.ndim
        if broadcastable is not None and variable is not None:
            assert broadcastable == variable.broadcastable
        if is_shared is not None and variable is not None:
            assert is_shared == isinstance(self.variable,
                                           theano.compile.SharedVariable)
        if dtype is not None and variable is not None:
            assert dtype == variable.dtype
        if tags is not None:
            self.verify_tags(set(tags))

    def verify_tags(self, tags):
        for tag in tags:
            assert tag in VALID_TAGS
        if self.is_shared:
            # only one of parameter and state should be set
            assert ("parameter" in tags) != ("state" in tags)
            if "parameter" in tags:
                # only one of weight and bias should be set
                assert ("weight" in tags) != ("bias" in tags)
            # the only valid tags for shared are the following:
            assert len(tags - {"weight", "bias", "parameter", "state"}) == 0
        else:
            assert len({"weight", "bias", "parameter", "state"} & tags) == 0

    @property
    def is_shared(self):
        if self.is_shared_ is None:
            # if is_shared is not supplied, a variable must be supplied
            assert self.variable_ is not None
            self.is_shared_ = isinstance(self.variable,
                                         theano.compile.SharedVariable)
        return self.is_shared_

    @property
    def tags(self):
        if self.tags_ is None:
            self.tags_ = []
        if not isinstance(self.tags_, set):
            self.tags_ = set(self.tags_)
        self.verify_tags(self.tags_)
        return self.tags_

    @property
    def ndim(self):
        if self.shape_ is not None:
            self.ndim_ = len(self.shape_)
        assert self.ndim_ is not None
        return self.ndim_

    @property
    def dtype(self):
        if self.dtype_ is None:
            self.dtype_ = floatX
        return self.dtype_

    @property
    def broadcastable(self):
        if self.broadcastable_ is None:
            self.broadcastable_ = (False, ) * self.ndim
        return self.broadcastable_

    @property
    def variable(self):
        if self.variable_ is None:
            if self.is_shared:
                # find appropriate initialization scheme
                shared_initializations = self.shared_initializations
                if shared_initializations is None:
                    shared_initializations = []
                for initialization in shared_initializations:
                    if initialization.predicate(self):
                        break
                else:
                    # default to zero initialization if none work
                    initialization = ZeroInitialization()

                # create the shared variable
                variable = initialization.create_shared(self)
            else:
                variable = T.TensorType(self.dtype,
                                        self.broadcastable)(self.name)
            self.variable_ = variable

            # for ease of debugging, add test values
            # ---
            # this must be done after self.variable_ is set to avoid a
            # recursive loop when calling self.shape
            if (not self.is_shared) and ENABLE_TEST_VALUE:
                test_value = np.random.rand(*self.shape).astype(self.dtype)
                variable.tag.test_value = test_value

        # add current variable to variable map
        if self.variable_map is not None:
            if self.variable_ in self.variable_map:
                assert self.variable_map[self.variable_] is self
            else:
                self.variable_map[self.variable_] = self

        return self.variable_

    @property
    def shape(self):
        if self.shape_ is None:
            # cannot derive shape for shared variable
            assert not self.is_shared

            # FIXME calculate shape
            assert False
        return self.shape_

    @property
    def value(self):
        assert self.is_shared
        return self.variable.get_value()

    @value.setter
    def value(self, new_value):
        assert new_value.dtype == self.dtype
        assert new_value.shape == self.shape
        self.variable.set_value(new_value)

i = T.iscalar()
o = LazyWrappedVariable("foo", variable=i).variable
fn = theano.function([i], o)
for _ in range(10):
    x = np.random.randint(1e6)
    assert fn(x) == x

s = LazyWrappedVariable("foo", shape=(3, 4, 5), is_shared=True)
assert s.value.sum() == 0
x = np.random.randn(3, 4, 5)
s.value = x
assert np.allclose(s.value, x)
try:
    s.value = np.random.randn(5, 4, 3)
except:
    pass
else:
    assert False


# TODO move to graph module


def init_name_to_node(root_node):
    # DFS traversal
    name_to_node = {}
    nodes = [root_node]
    while nodes:
        node = nodes.pop()
        name = node.name
        # make sure that all names are unique
        assert name not in name_to_node
        name_to_node[name] = node

        # add children to stack
        children = node.architecture_children()
        nodes.extend(children)
    return name_to_node


def init_architectural_tree(name_to_node):
    g = nx.DiGraph()
    # DFS traversal
    for node in name_to_node.values():
        name = node.name
        # need to manually add the node, because no nodes are added in the
        # case of a single node graph
        g.add_node(name)
        children = node.architecture_children()
        for child in children:
            # set edge from child to parent to reflect dependency of parent
            # on child
            g.add_edge(child.name, name)
    assert (set(name_to_node.keys()) == set(g.nodes()))
    return g


class TreeanoGraph(object):

    """
    class that stores all the graph state for a network of nodes, including
    1. architectural tree
    2. computation graph dependencies

    a single mutable graph shared between all the nodes in a network
    """

    def __init__(self, root_node):
        self.name_to_node = init_name_to_node(root_node)
        self.architectural_tree = init_architectural_tree(self.name_to_node)
        # since all parents automatically depend on their children,
        # we can initialize the computation grpah as a copy of the
        # architectural tree
        self.computation_graph = self.architectural_tree.copy()

    def nodes(self, order=None):
        """
        returns all nodes in the graph
        """
        if order is None:
            node_names = self.name_to_node.keys()
        elif order == "architecture":
            node_names = nx.topological_sort(self.architectural_tree)
        elif order == "computation":
            node_names = nx.topological_sort(self.computation_graph)
        else:
            raise ValueError("Unknown order: %s" % order)
        # make sure that all of the original nodes are returned
        assert set(self.name_to_node.keys()) == set(node_names)
        return [self.name_to_node[name] for name in node_names]

    def add_dependency(self,
                       from_name,
                       to_name,
                       from_key="default",
                       to_key="default"):
        """
        adds a computation graph dependency from a node with "from_name" as a
        name to a node with "to_name" as a name

        from_key: key in output of from-node to be passed into to-node

        to_key: key to-node will use to query the dependency - each input to
        a node must have a unique to_key
        """
        # make sure that these nodes are part of this graph
        assert from_name in self.name_to_node
        assert to_name in self.name_to_node
        # make sure that to_key is unique for to-node
        for _, edge_to, datamap in self.computation_graph.edges(data=True):
            if edge_to == to_name and datamap.get("to_key") == to_key:
                raise ValueError("Non-unique to_key(%s) found for node %s"
                                 % (to_key, to_name))
        # add the dependency
        self.computation_graph.add_edge(from_name,
                                        to_name,
                                        from_key=from_key,
                                        to_key=to_key)
        # make sure that the dependency doesn't cause any cycles
        try:
            nx.topological_sort(self.computation_graph)
        except nx.NetworkXUnfeasible:
            self.computation_graph.remove_edge(from_name, to_name)
            # TODO maybe use a custom exception
            raise

    def input_edge_for_node(self, node_name, to_key="default"):
        """
        searches for the input node and from_key of a given node with a given
        to_key, and returns None if not found
        """
        edges = self.computation_graph.edges(data=True)
        for edge_from, edge_to, datamap in edges:
            if edge_to == node_name and datamap.get("to_key") == to_key:
                from_key = datamap["from_key"]
                return (edge_from, from_key)
        return None

    def input_for_node(self, node_name, to_key="default"):
        """
        searches for an input to a node with a given to_key
        """
        edge_from, from_key = self.input_edge_for_node(node_name, to_key)
        node = self.name_to_node[edge_from]
        return node.output[from_key]

    def output_edge_for_node(self, node_name, from_key="default"):
        """
        searches for the node and from_key, and returns None if not found

        this is a trivial operation, made to parallel input_edge_for_node
        """
        node = self.name_to_node[node_name]

        if from_key in node.output:
            return (node_name, from_key)
        else:
            return None

    def output_for_node(self, node_name, from_key="default"):
        """
        returns the output of a node with the given key
        """
        node = self.name_to_node[node_name]
        return node.output[from_key]

    def architecture_ancestor_names(self, node_name):
        """
        returns a generator of ancestor names of the current node in the
        architectural tree, in the order of being closer to the node
        towards the root
        """
        current_name = node_name
        while True:
            current_parents = self.architectural_tree.successors(current_name)
            if len(current_parents) == 0:
                break
            elif len(current_parents) > 1:
                # in a tree, each node should have a single parent, except
                # the root
                assert False
            else:
                current_name, = current_parents
                yield current_name

    def architecture_ancestors(self, node_name):
        """
        returns a generator of ancestors of the current node in the
        architectural tree, in the order of being closer to the node
        towards the root
        """
        for ancestor_name in self.architecture_ancestor_names(node_name):
            yield self.name_to_node[ancestor_name]

    def architecture_subtree_names(self, node_name):
        """
        returns an unordered set of descendant names of the current node in the
        architectural tree
        """
        # NOTE: this is actually the descendants, despite the name, because
        # of the way we set up the tree
        descendant_names = nx.ancestors(self.architectural_tree, node_name)
        subtree_names = descendant_names | {node_name}
        return subtree_names

    def architecture_subtree(self, node_name):
        """
        returns an unordered set of descendants of the current node in the
        architectural tree
        """
        return {self.name_to_node[name]
                for name in self.architecture_subtree_names(node_name)}


# TODO move to node module


class Node(object):

    """
    all nodes require a unique name attribute as a string
    """

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
        return root_node

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
            all_deltas = UpdateDeltas({})
            for node in self.graph.nodes(order="computation"):
                if not hasattr(node, "update_deltas"):
                    update_deltas = node.compute_update_deltas()
                    node.update_deltas = update_deltas
                all_deltas += node.update_deltas

            # combine with manually specified updates
            if updates is not None:
                update_deltas = UpdateDeltas.from_updates(updates)
                all_deltas += update_deltas

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

    def find_variables(self, tag_filters):
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

    def compute_update_deltas(self):
        """
        computes updates of a node as UpdateDeltas

        optional to define - if the node doesn't update itself
        """
        return UpdateDeltas({})


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


class ReferenceNode(Fields.name.reference, Node):

    """
    provides dependencies into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them
    """

    def init_state(self):
        self.graph.add_dependency(self.reference, self.name)

    def compute_output(self):
        return dict(
            default=self.get_input()
        )


class SequentialNode(Fields.name.nodes, WrapperNode):

    """
    applies several nodes sequentially
    """

    def architecture_children(self):
        return self.nodes

    def init_state(self):
        children_names = self.children_names()
        for from_name, to_name in zip(children_names,
                                      children_names[1:]):
            self.graph.add_dependency(from_name, to_name)
        # set input of first child as default input of this node
        self.forward_input_to(children_names[0])
        # set input of this node as output of final child
        self.take_input_from(children_names[-1])


class ContainerNode(Fields.name.nodes, WrapperNode):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    def architecture_children(self):
        return self.nodes

    def init_state(self):
        # by default, returns the output of its first child
        # ---
        # this was done because it's a sensible default, and other nodes
        # assume that every node has an output
        # additionally, returning the input of this node didn't work, because
        # sometimes the node has no input (eg. if it contains the input node)
        self.take_input_from(self.nodes[0].name)


class HyperparameterNode(Fields.name.node.hyperparameters, WrapperNode):

    """
    for providing hyperparameters to a subtree
    """

    def __init__(self, name, node, **hyperparameters):
        # override init to allow for using keyword arguments
        super(HyperparameterNode, self).__init__(name, node, hyperparameters)

    def architecture_children(self):
        return [self.node]

    def get_hyperparameter(self, hyperparameter_name):
        return self.hyperparameters.get(hyperparameter_name, None)

    def init_state(self):
        self.forward_input_to(self.node.name)
        self.take_input_from(self.node.name)


class InputNode(Fields.name.shape.dtype[floatX].broadcastable[None], Node):

    """
    an entry point into the network
    """

    def compute_output(self):
        self.create_variable(
            name="input_var",
            shape=self.shape,
            dtype=self.dtype,
            broadcastable=self.broadcastable,
            is_shared=False,
            tags=["input"],
        )
        return dict(
            default=self.input_var,
        )


class IdentityNode(Fields.name, Node):

    """
    returns input
    """

    def compute_output(self):
        return dict(
            default=self.get_input()
        )


class FullyConnectedNode(Fields.name.num_units[None], Node):

    """
    node wrapping lasagne's DenseLayer
    """

    def compute_output(self):
        in_var = self.get_input()
        num_units = self.find_hyperparameter("num_units")
        num_inputs = int(np.prod(in_var.shape[1:]))
        self.create_variable(
            "W",
            shape=(num_inputs, num_units),
            is_shared=True,
            tags=["parameter", "weight"]
        )
        self.create_variable(
            "b",
            shape=(num_units,),
            is_shared=True,
            tags=["parameter", "bias"]
        )
        self.input_layer = lasagne.layers.InputLayer(
            in_var.shape
        )
        self.dense_layer = lasagne.layers.DenseLayer(
            incoming=self.input_layer,
            num_units=num_units,
            W=self.W.variable,
            b=self.b.variable,
            nonlinearity=lasagne.nonlinearities.identity,
        )
        out_variable = self.dense_layer.get_output_for(in_var.variable)
        out_shape = self.dense_layer.get_output_shape_for(in_var.shape)
        self.create_variable(
            "result",
            variable=out_variable,
            shape=out_shape,
        )
        return dict(
            default=self.result,
        )


class ReLUNode(Fields.name, Node):

    """
    rectified linear unit
    """

    def compute_output(self):
        input = self.get_input()
        in_variable = input.variable
        out_variable = lasagne.nonlinearities.rectify(in_variable)
        self.create_variable(
            "result",
            variable=out_variable,
            shape=input.shape,
        )
        return dict(
            default=self.result,
        )


LOSS_AGGREGATORS = {
    'mean': T.mean,
    'sum': T.sum,
}


class CostNode(Fields.name.target_reference.loss_function.loss_aggregator,
               Node):

    """
    takes in a loss function and a reference to a target node, and computes
    the aggregate loss between the nodes input and the target
    """

    def __init__(self,
                 name,
                 target_reference,
                 loss_function=None,
                 loss_aggregator=None):
        super(CostNode, self).__init__(name,
                                       target_reference,
                                       loss_function,
                                       loss_aggregator)

    def init_state(self):
        self.graph.add_dependency(self.target_reference,
                                  self.name,
                                  to_key="target")

    def compute_output(self):
        preds = self.get_input().variable
        target = self.get_input(to_key="target").variable

        loss_function = self.find_hyperparameter("loss_function")
        self.cost = loss_function(preds, target)

        loss_aggregator = self.find_hyperparameter("loss_aggregator", "mean")
        loss_aggregator = LOSS_AGGREGATORS.get(loss_aggregator,
                                               # allow user defined function
                                               loss_aggregator)
        self.aggregate_cost = loss_aggregator(self.cost)

        self.create_variable(
            "result",
            variable=self.aggregate_cost,
            shape=(),
        )
        return dict(
            default=self.result,
        )


# def test node_constructor_arguments():


class foo(Fields.a.b.c, Node):
    pass

assert foo.constructor_arguments() == ["a", "b", "c"]


# def test_node_to_from_architecture_data():
class foo(Fields.a.b.c, Node):
    pass

f = foo(3, 4, 5)
assert f == f.__class__.from_architecture_data(f.to_architecture_data())
assert f == f.from_architecture_data(f.to_architecture_data())


# def test_architecture_test_node_copy():
class foo(Fields.a.b.c, Node):
    pass

f = foo(3, 4, 5)
assert f == f.architecture_copy()

# def test_identity_network():
input_node = InputNode("foo", (3, 4, 5))
network = input_node.build()
fn = network.function(["foo"], ["foo"])
x = np.random.rand(3, 4, 5).astype(floatX)
assert np.allclose(fn(x), x)

# def test_sequential_identity_network():
nodes = [
    InputNode("foo", (3, 4, 5)),
    IdentityNode("bar"),
]
sequential = SequentialNode("choo", nodes)
network = sequential.build()
fn1 = network.function(["foo"], ["foo"])
fn2 = network.function(["foo"], ["bar"])
fn3 = network.function(["foo"], ["choo"])
x = np.random.rand(3, 4, 5).astype(floatX)
assert np.allclose(fn1(x), x)
assert np.allclose(fn2(x), x)
assert np.allclose(fn3(x), x)

# def test_nested_sequential_network():
current_node = InputNode("foo", (3, 4, 5))
for name in map(str, range(10)):
    current_node = SequentialNode("sequential" + name,
                                  [current_node,
                                   IdentityNode("identity" + name)])
network = current_node.build()
fn = network.function(["foo"], ["sequential9"])
x = np.random.rand(3, 4, 5).astype(floatX)
assert np.allclose(fn(x), x)
if False:
    # NOTE: ugly
    import pylab
    nx.draw_networkx(
        network.graph.computation_graph,
        nx.spring_layout(network.graph.computation_graph),
        node_size=500)
    pylab.show()
if False:
    # plot computation_graph
    import pylab
    nx.draw_networkx(
        network.graph.computation_graph,
        nx.graphviz_layout(network.graph.computation_graph),
        node_size=500)
    pylab.show()
if False:
    # plot architectural_tree
    import pylab
    nx.draw_networkx(
        network.graph.architectural_tree,
        nx.graphviz_layout(network.graph.architectural_tree),
        node_size=500)
    pylab.show()

# def test_toy_updater_node():


class ToyUpdaterNode(Fields.name, Node):

    """
    example node to test compute_update_deltas
    """

    def compute_output(self):
        shape = (2, 3, 4)
        self.create_variable(
            name="state",
            shape=shape,
            is_shared=True,
            tags=["state"],
        )
        init_value = np.arange(np.prod(shape)).reshape(*shape).astype(floatX)
        self.state.value = init_value
        return dict(
            default=self.state
        )

    def compute_update_deltas(self):
        return UpdateDeltas({
            self.state.variable: 42
        })


network = ToyUpdaterNode("a").build()
fn1 = network.function([], ["a"])
init_value = fn1()
fn2 = network.function([], ["a"], generate_updates=True)
assert np.allclose(init_value, fn2())
assert np.allclose(init_value[0] + 42, fn2())
assert np.allclose(init_value[0] + 84, fn1())
assert np.allclose(init_value[0] + 84, fn1())


# def test_hyperparameter_node():
input_node = InputNode("a", (3, 4, 5))
hp_node = HyperparameterNode("b", input_node, foo=3, bar=2)
network = hp_node.build()
assert network.get_hyperparameter("foo") == 3
a_node = network.graph.name_to_node["a"]
assert a_node.get_hyperparameter("foo") is None
assert a_node.find_hyperparameter("foo") == 3


# def test_ones_initialization():

class OnesInitialization(SharedInitialization):

    def initialize_value(self, var):
        return np.ones(var.shape).astype(var.dtype)


class DummyNode(Node):

    def __init__(self):
        self.name = "dummy"

    def get_hyperparameter(self, hyperparameter_name):
        if hyperparameter_name == "shared_initializations":
            return [OnesInitialization()]

    def compute_output(self):
        self.create_variable(
            "foo",
            is_shared=True,
            shape=(1, 2, 3),
        )
        return dict(
            default=self.foo,
        )

network = DummyNode().build()
fn = network.function([], ["dummy"])
assert np.allclose(fn(), np.ones((1, 2, 3)).astype(floatX))

# def test_fully_connected_node():
nodes = [
    InputNode("a", (3, 4, 5)),
    FullyConnectedNode("b"),
]
sequential = SequentialNode("c", nodes)
hp_node = HyperparameterNode("d",
                             sequential,
                             num_units=14,
                             shared_initializations=[OnesInitialization()])
network = hp_node.build()
fn = network.function(["a"], ["d"])
x = np.random.randn(3, 4, 5).astype(floatX)
res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
assert np.allclose(fn(x), res)

# def test_fully_connected_and_relu_node():
nodes = [
    InputNode("a", (3, 4, 5)),
    FullyConnectedNode("b"),
    ReLUNode("e"),
]
sequential = SequentialNode("c", nodes)
hp_node = HyperparameterNode("d",
                             sequential,
                             num_units=14,
                             shared_initializations=[OnesInitialization()])
network = hp_node.build()
fn = network.function(["a"], ["d"])
x = np.random.randn(3, 4, 5).astype(floatX)
res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
assert np.allclose(fn(x), np.clip(res, 0, np.inf))

# def test_glorot_uniform_initialization():
nodes = [
    InputNode("a", (3, 4, 5)),
    FullyConnectedNode("b"),
    ReLUNode("e"),
]
sequential = SequentialNode("c", nodes)
hp_node = HyperparameterNode("d",
                             sequential,
                             num_units=1000,
                             shared_initializations=[GlorotUniform()])
network = hp_node.build()
fc_node = network.node.nodes[1]
W_value = fc_node.W.value
assert np.allclose(0, W_value.mean(), atol=1e-2)
assert np.allclose(np.sqrt(2.0 / (20 + 1000)), W_value.std(), atol=1e-2)
assert np.allclose(np.zeros(1000), fc_node.b.value)


# def test_cost_node():
network = HyperparameterNode(
    "g",
    ContainerNode("f", [
        SequentialNode("e", [
            InputNode("input", (3, 4, 5)),
            FullyConnectedNode("b"),
            ReLUNode("c"),
            CostNode("cost", "target"),
        ]),
        InputNode("target", (3, 14)),
    ]),
    num_units=14,
    loss_function=lasagne.objectives.mse,
    shared_initializations=[OnesInitialization()]
).build()
fn = network.function(["input", "target"], ["cost"])
x = np.random.randn(3, 4, 5).astype(floatX)
res = np.dot(x.reshape(3, 20), np.ones((20, 14))) + np.ones(14)
res = np.clip(res, 0, np.inf)
y = np.random.randn(3, 14).astype(floatX)
res = np.mean((y - res) ** 2)
assert np.allclose(fn(x, y), res)
