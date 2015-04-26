from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import inspect
from fields import Fields
import networkx as nx

# TODO move everything under treeano.core

# TODO move to variable module


class LazyVariable(object):
    pass

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
        # making architectural tree immutable - hopefully this doesn't
        # hurt flexibility
        self.architectural_tree = self.architectural_tree.freeze()
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
            if edge_to == to_name and datamap["to_key"] == to_key:
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

    def freeze(self):
        """
        prevents mutation of internal graph
        """
        self.computation_graph = self.computation_graph.freeze()

    def input_edge_for_node(self, node_name, to_key="default"):
        """
        searches for the input node and from_key of a given node with a given
        to_key
        """
        edges = self.computation_graph.edges(data=True)
        for edge_from, edge_to, datamap in edges:
            if edge_to == node_name and datamap["to_key"] == to_key:
                from_key = datamap["from_key"]
                return (edge_from, from_key)
        raise ValueError("Input with to_key %s not found for node %s"
                         % (to_key, node_name))

    def input_for_node(self, node_name, to_key="default"):
        """
        searches for an input to a node with a given to_key
        """
        edge_from, from_key = self.input_edge_for_node(node_name, to_key)
        node = self.name_to_node[edge_from]
        return node.output[from_key]


# TODO move to node module


class Node(object):

    """
    all nodes require a unique name attribute
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
        for node in graph.nodes(order="architecture"):
            # share graph between all nodes
            node.graph = graph
            # recursively init nodes
            node.init_state()
        # make graph immutable
        graph.freeze()
        # compute and cache outputs
        # ---
        # this is reversed so that outer nodes have their state initialized
        # before inner nodes - this is important for sequential nodes, since
        # the first child will depend on the input of the sequential node, and
        # we would like to make that dependency explicit
        for node in reversed(graph.nodes(order="computation")):
            output = node.compute()
            node.output = output
        return root_node

    def get_input(self, to_key="default"):
        """
        returns input of the current node in the graph for a given key
        """
        return self.graph.input_for_node(self.name, to_key)

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

    def compute(self):
        """
        computes output of a node
        """
        raise NotImplementedError


class ReferenceNode(Fields.name.reference, Node):

    """
    provides dependencies into separate parts of the tree, allowing
    separate branches to have computational graph dependencies between them
    """

    def init_state(self):
        self.graph.add_dependency(self.reference, self.name)

    def compute(self):
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
        name_from, from_key = self.graph.input_edge_for_node(self.name)
        self.graph.add_dependency(name_from,
                                  children_names[0],
                                  from_key=from_key)
        # set input of this node as output of final child
        self.graph.add_dependency(children_names[-1],
                                  self.name,
                                  to_key="last_child")

    def compute(self):
        """
        returns output of final child
        """
        return dict(
            default=self.get_input(to_key="last_child")
        )


class InputNode(Fields.name, Node):
    # FIXME
    pass


class ContainerNode(Fields.name.nodes, Node):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    def architecture_children(self):
        return self.nodes

    def compute(self):
        # TODO add an input index and an output index
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


# def architecture_test_node_copy():
class foo(Fields.a.b.c, Node):
    pass

f = foo(3, 4, 5)
assert f == f.architecture_copy()
