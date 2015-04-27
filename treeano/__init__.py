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

floatX = theano.config.floatX
ENABLE_TEST_VALUE = theano.config.compute_test_value != "off"

# TODO move everything under treeano.core

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
                 shape=None,
                 dtype=None,
                 broadcastable=None,
                 is_shared=None,
                 tags=None,
                 ndim=None,
                 variable=None,):
        self.name = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.broadcastable_ = broadcastable
        self.is_shared_ = is_shared
        self.tags_ = tags
        self.ndim_ = ndim
        self.variable_ = variable

    def verify_tags(self):
        tags = set(self.tags)
        for tag in tags:
            assert tag in VALID_TAGS
        if self.is_shared:
            # only one of parameter and state should be set
            assert ("parameter" in tags) != ("state" in tags)
            if "parameter" in tags:
                # only one of weight and bias should be set
                assert ("weight" in tags) != ("bias" in tags)
            # the only valid tags for shared are the following:
            assert len({"weight", "bias", "parameter", "state"} - tags) == 0
        else:
            assert len({"weight", "bias", "parameter", "state"} & tags) == 0

    @property
    def is_shared(self):
        if self.is_shared_ is None:
            # if is_shared is not supplied, a variable must be supplied
            assert self.variable is not None
            self.is_shared_ = isinstance(self.variable,
                                         theano.compile.SharedVariable)
        return self.is_shared_

    @property
    def tags(self):
        if self.tags_ is None:
            self.tags_ = []
        self.verify_tags()
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
                kwargs = {}
                if len(self.broadcastable) > 0:
                    kwargs["broadcastable"] = self.broadcastable
                    value = np.zeros(self.shape)
                else:
                    value = 0
                kwargs["name"] = self.name
                kwargs["value"] = np.array(value).astype(self.dtype)
                variable = theano.shared(**kwargs)
            else:
                variable = T.TensorType(self.dtype,
                                        self.broadcastable)(self.name)
            # add current variable
            variable.tag.lazy_wrapped_variable = self
            self.variable_ = variable

            # for ease of debugging, add test values
            # ---
            # this must be done after self.variable_ is set to avoid a
            # recursive loop when calling self.shape
            if ENABLE_TEST_VALUE:
                test_value = np.random.rand(*self.shape).astype(self.dtype)
                variable.tag.test_value = test_value
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
        to_key
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

    def output_for_node(self, node_name, from_key="default"):
        """
        returns the output of a node with the given key
        """
        node = self.name_to_node[node_name]
        return node.output[from_key]

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
        # create the variable
        variable = LazyWrappedVariable(new_name, *args, **kwargs)
        # set variable as attribute for easy access
        setattr(self, name, variable)
        # register variable for future searching of parameters
        if not hasattr(self, "variables"):
            self.variables = []
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

    def find_parameters(self, ):
        # TODO fill in arguments (filters based on tag?)
        return []

    def get_hyperparameter(self, hyperparameter_name):
        # FIXME recursively walk up graph
        # for each node: self.graph = computation_graph
        pass

    def get_parameters(self):
        """
        returns all parameters of a node

        optional to define (eg. if node has no parameters)
        """
        return []

    def architecture_children(self):
        """
        returns all child nodes of the given node
        """
        # TODO maybe allow for a filter here by node type
        # TODO maybe have a composite node base class which throws an
        # exception here
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


class SequentialNode(Fields.name.nodes, Node):

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
        input_edge = self.graph.input_edge_for_node(self.name)
        # there may not be an input (eg. if the sequential node is holding
        # the input node)
        if input_edge is not None:
            name_from, from_key = input_edge
            self.graph.add_dependency(name_from,
                                      children_names[0],
                                      from_key=from_key)
        # set input of this node as output of final child
        self.graph.add_dependency(children_names[-1],
                                  self.name,
                                  to_key="last_child")

    def compute_output(self):
        """
        returns output of final child
        """
        return dict(
            default=self.get_input(to_key="last_child")
        )


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


class ContainerNode(Fields.name.nodes, Node):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    def architecture_children(self):
        return self.nodes

    def compute_output(self):
        pass


class HyperparameterNode(Fields.name.node.hyperparameters, Node):

    """
    for providing hyperparameters to a subtree
    """

    # TODO


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
