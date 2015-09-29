"""
tree based transformations
"""

import treeano
import treeano.nodes as tn

from . import fns
from .. import node_utils


def remove_nodes(network, names_to_remove, keep_child=False, **kwargs):
    """
    replaces nodes with the given names with the node's single child if it
    has children or with an identitynode if the node doesn't have children
    """
    def inner(node):
        if node.name in names_to_remove:
            if keep_child:
                children = node.architecture_children()
                assert len(children) == 1
                return children[0]
            else:
                return tn.IdentityNode(node.name)
        else:
            return node

    return fns.transform_root_node_postwalk(network, inner, **kwargs)


def remove_subtree(network, names_to_remove, **kwargs):
    """
    replaces entire subtree of nodes with the given names with IdentityNode's
    """
    def inner(node):
        if node.name in names_to_remove:
            return tn.IdentityNode(node.name)
        else:
            return node

    return fns.transform_root_node_postwalk(network, inner, **kwargs)


def remove_parent(network, names, **kwargs):
    """
    replaces parents of the given nodes with the node (removes all other
    children)
    """
    mutable_names = set(names)

    def inner(node):
        for child in node.architecture_children():
            if child.name in mutable_names:
                # need to remove name from set because we are postwalking
                # and the same property would apply to the node's new parent
                mutable_names.remove(child.name)
                return child
        return node

    res = fns.transform_root_node_postwalk(network, inner, **kwargs)
    # all of the given names should be part of the network
    assert len(mutable_names) == 0, mutable_names
    return res


def add_parent(network,
               name,
               parent_constructor,
               parent_name,
               parent_kwargs,
               **kwargs):
    """
    adds a parent between the given node and its current parent

    name:
    name of the node to add the parent to
    """
    found = [False]

    def inner(node):
        if node.name == name:
            found[0] = True
            return parent_constructor(
                parent_name,
                node,
                **parent_kwargs
            )
        else:
            return node

    res = fns.transform_root_node_postwalk(network, inner, **kwargs)
    assert found[0], "%s not found" % name
    return res


def add_hyperparameters(network, name, hyperparameters, **kwargs):
    """
    adds a new root hyperparameter node with the given name and hyperparameters
    """
    def inner(root_node):
        return tn.HyperparameterNode(
            name,
            root_node,
            **hyperparameters
        )

    return fns.transform_root_node(network, inner, **kwargs)


def remove_parents(network, name, **kwargs):
    """
    removes all parents of the given node name, replacing the root node with
    itself
    """

    def inner(root_node):
        if network.is_built:
            graph = network.graph
        else:
            graph = treeano.core.graph.TreeanoGraph(root_node)
        return graph.name_to_node[name]

    return fns.transform_root_node(network, inner, **kwargs)


def move_node(network, new_name, old_name, **kwargs):
    """
    moves a given node into the location in the tree of another node
    """
    def inner(root_node):
        if network.is_built:
            graph = network.graph
        else:
            graph = treeano.core.graph.TreeanoGraph(root_node)
        new_node = graph.name_to_node[new_name]

        def fn(node):
            if node.name == old_name:
                return new_node
            else:
                return node

        return node_utils.postwalk_node(root_node, fn)

    return fns.transform_root_node(network, inner, **kwargs)
