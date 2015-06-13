import treeano

from .. import network_utils
from .. import walk_utils


def transform_root_node(network, fn, priority="post_override"):
    """
    takes in a function that manipulates a node tree and returns a transformed
    network

    priority:
    one of:
    - pre_override = add variable initialization to the beginning of
                     override_hyperparameters
    - post_override = add variable initialization to the end of
                      override_hyperparameters
    - pre_default = add variable initialization to the beginning of
                    default_hyperparameters
    - post_default = add variable initialization to the end of
                     default_hyperparameters
    """
    assert priority in {"pre_override",
                        "post_override",
                        "pre_default",
                        "post_default"}

    root_node = network.root_node
    override_hyperparameters = network.override_hyperparameters
    default_hyperparameters = network.default_hyperparameters

    new_root_node = fn(root_node)

    if network.is_built:
        if priority.endswith("override"):
            init_map = override_hyperparameters
        else:
            # priority ends with default
            init_map = default_hyperparameters

        inits = init_map.get("inits", [])
        init_map["inits"] = inits
        # TODO flag on whether or not to share variables or just initialize
        # with values
        preallocated_init = network_utils.to_preallocated_init(network)

        if priority.startswith("pre"):
            inits.insert(0, preallocated_init)
        else:
            # priority starts with post
            inits.append(preallocated_init)

    return treeano.Network(
        root_node=new_root_node,
        override_hyperparameters=override_hyperparameters,
        default_hyperparameters=default_hyperparameters,
    )


def transform_node_data(network, fn, **kwargs):
    """
    takes in a function that manipulates a node as data and returns a
    transformed network
    """
    def inner(root_node):
        as_data = treeano.core.node_to_data(root_node)
        transformed = fn(as_data)
        as_node = treeano.core.node_from_data(transformed)
        return as_node

    return transform_root_node(network, inner, **kwargs)


def transform_root_node_postwalk(network, fn, **kwargs):
    """
    takes in a function that manipulates a node as data that is applied
    in a postwalk (ie. leaves first) to all nodes in a tree, and returns a
    transformed network
    """
    def postwalk_fn(obj):
        if isinstance(obj, treeano.core.NodeAPI):
            return fn(obj)
        else:
            return obj

    def inner(root_node):
        return walk_utils.walk(root_node, postwalk_fn=postwalk_fn)

    return transform_root_node(network, inner, **kwargs)


def transform_node_data_postwalk(network, fn, **kwargs):
    """
    takes in a function that manipulates a node as data that is applied
    in a postwalk (ie. leaves first) to all nodes in a tree, and returns a
    transformed network
    """
    def inner(data):
        return walk_utils.collection_postwalk(data, postwalk_fn=fn)

    return transform_node_data(network, inner, **kwargs)
