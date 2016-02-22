from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import networkx as nx


def init_name_to_node(root_node):
    # DFS traversal
    name_to_node = {}
    nodes = [root_node]
    while nodes:
        node = nodes.pop()
        name = node.name
        # make sure that all names are unique
        assert name not in name_to_node, dict(
            name=name,
            prev_node=name_to_node[name],
            curr_node=node,
        )
        name_to_node[name] = node

        # add children to stack
        children = node.architecture_children()
        for child in children:
            assert child.name
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
        # - since all parents automatically depend on their children,
        #   we can initialize the computation grpah as a copy of the
        #   architectural tree
        # - this is a multi graph so the same node can be the input to a given
        #   node multiple times
        self.computation_graph = nx.MultiDiGraph(
            self.architectural_tree.copy())
        self.is_mutable = True

    def _nodes(self, order=None):
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

    def architectural_tree_nodes_leaves_to_root(self):
        return self._nodes("architecture")

    def architectural_tree_nodes_root_to_leaves(self):
        return reversed(self._nodes("architecture"))

    def computation_graph_nodes_topological(self):
        return self._nodes("computation")

    def nodes(self):
        return self._nodes(None)

    def remove_dependency(self, from_name, to_name):
        """
        removes a computation graph dependency from a node with "from_name" as
        a name to a node with "to_name" as a name
        """
        self.computation_graph.remove_edge(from_name, to_name)

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
        # make sure network is mutable
        assert self.is_mutable
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
            # TODO this might not be sufficient, since an edge between
            # from_name to to_name might have existed before this operation
            self.computation_graph.remove_edge(from_name, to_name)
            # TODO maybe use a custom exception
            raise

    def all_input_edges_for_node(self, node_name):
        """
        returns all edges and their corresponding data going into the given
        node
        """
        edges = self.computation_graph.edges(data=True)
        for edge_from, edge_to, datamap in edges:
            if edge_to == node_name:
                yield (edge_from, edge_to, datamap)

    def input_edge_for_node(self, node_name, to_key="default"):
        """
        searches for the input node and from_key of a given node with a given
        to_key, and returns None if not found
        """
        for edge_from, _, datamap in self.all_input_edges_for_node(node_name):
            if datamap.get("to_key") == to_key:
                from_key = datamap["from_key"]
                return (edge_from, from_key)
        return None

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
