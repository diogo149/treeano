"""
tree based transformations
"""

import treeano.nodes as tn

from . import fns


def remove_node(network, names_to_remove, **kwargs):
    """
    replaces nodes with the given names with the node's single child
    """
    def inner(node):
        if node.name in names_to_remove:
            children = node.architecture_children()
            assert len(children) == 1
            return children[0]
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
    def inner(node):
        for child in node.architecture_children():
            if child in names:
                return child
        return node

    return fns.transform_root_node_postwalk(network, inner, **kwargs)


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
