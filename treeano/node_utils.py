from . import core


def copy_node(node):
    return core.node_from_data(core.node_to_data(node))
