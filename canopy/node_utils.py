"""
TODO should this be in treeano.node_utils
"""

import treeano

from . import walk_utils


def postwalk_node(root_node, fn):
    """
    traverses a tree of nodes in a postwalk with a function that can
    transform nodes
    """
    def postwalk_fn(obj):
        if isinstance(obj, treeano.core.NodeAPI):
            res = fn(obj)
            assert isinstance(res, treeano.core.NodeAPI)
            return res
        else:
            return obj

    return walk_utils.walk(root_node, postwalk_fn=postwalk_fn)


def suffix_node(root_node, suffix):
    """
    creates a copy of a node, with names suffixed by given suffix
    """
    # use seen set to make sure there are no bugs
    seen = set()

    def copy_and_suffix(node):
        assert node.name not in seen
        seen.add(node.name)
        new_node = treeano.node_utils.copy_node(node)
        # assert that node is nodeimpl, since we only know how to set
        # name for those
        assert isinstance(new_node, treeano.NodeImpl)
        new_node._name += suffix
        return new_node

    return postwalk_node(root_node, copy_and_suffix)
