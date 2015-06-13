import treeano


def copy_node(node):
    return treeano.core.node_from_data(treeano.core.node_to_data(node))
