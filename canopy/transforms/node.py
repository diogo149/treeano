"""
node based transformations
"""

import treeano
import treeano.nodes as tn

from . import fns


def remove_dropout(network, **kwargs):
    """
    replaced DropoutNode's with IdentityNode's with the same name
    """

    def inner(node):
        if isinstance(node, tn.DropoutNode):
            return tn.IdentityNode(node.name)
        else:
            return node

    return fns.transform_root_node_postwalk(network, inner, **kwargs)


def replace_node(network, name_to_node, **kwargs):
    """
    name_to_node:
    map from name of the node to replace, to the new node
    """

    def inner(node):
        if node.name in name_to_node:
            return name_to_node[node.name]
        else:
            return node

    return fns.transform_root_node_postwalk(network, inner, **kwargs)


def update_hyperparameters(network, node_name, hyperparameters, **kwargs):
    """
    updates a node's hyperparameters
    """

    found = [False]

    def inner(node):
        if node.name == node_name:
            found[0] = True
            for k in hyperparameters:
                assert k in node.hyperparameter_names
            new_node = treeano.node_utils.copy_node(node)
            new_node.hyperparameters.update(hyperparameters)
            return new_node
        else:
            return node

    res = fns.transform_root_node_postwalk(network, inner, **kwargs)
    assert found[0], "%s not found in network" % node_name
    return res
