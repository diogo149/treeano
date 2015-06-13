"""
node based transformations
"""

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
