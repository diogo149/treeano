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
        # assert that node is nodeimpl, since we only know how to set
        # name for those
        assert isinstance(node, treeano.NodeImpl)
        node._name += suffix
        return node

    # make a copy only once here
    new_node = treeano.node_utils.copy_node(root_node)
    return postwalk_node(new_node, copy_and_suffix)


def format_node_name(root_node, format):
    """
    creates a copy of a node, with names suffixed by given suffix
    """
    # use seen set to make sure there are no bugs
    seen = set()

    def copy_and_format(node):
        assert node.name not in seen
        seen.add(node.name)
        # assert that node is nodeimpl, since we only know how to set
        # name for those
        assert isinstance(node, treeano.NodeImpl)
        node._name = format % node.name
        return node

    # make a copy only once here
    new_node = treeano.node_utils.copy_node(root_node)
    return postwalk_node(new_node, copy_and_format)
