import six

CHILDREN_CONTAINERS = {}
NODES = {}


def register_node(name):
    """
    registers the decorated node with the given string for serialization
    """
    assert isinstance(name, six.string_types)

    def inner(cls):
        # we want to allow overwriting (eg. if the file is refreshed)
        # sometimes, but not accidentaly overwriting
        if name in NODES:
            assert cls.__name__ == NODES[name].__name__
        NODES[name] = cls
        return cls
    return inner


def node_to_str(cls):
    """
    returns the string for the given registered node class
    """
    return {v: k for k, v in NODES.items()}[cls]


def node_from_str(s):
    """
    returns the registered node class for the given string
    """
    return NODES[s]


def node_to_data(node):
    """
    returns the given node as data
    """
    return dict(
        node_key=node_to_str(node.__class__),
        architecture_data=node._to_architecture_data(),
    )


def node_from_data(data):
    """
    convert the given node-representation-as-data back into an instance of
    the appropriate node class
    """
    node_key = data["node_key"]
    architecture_data = data["architecture_data"]
    return node_from_str(node_key)._from_architecture_data(architecture_data)


def register_children_container(name):
    """
    registers the decorated children container with the given string for
    serialization
    """
    assert isinstance(name, six.string_types)

    def inner(cls):
        # we want to allow overwriting (eg. if the file is refreshed)
        # sometimes, but not accidentaly overwriting
        if name in CHILDREN_CONTAINERS:
            assert cls.__name__ == CHILDREN_CONTAINERS[name].__name__
        CHILDREN_CONTAINERS[name] = cls
        return cls
    return inner


def children_container_to_str(cls):
    """
    returns the string for the given registered children container class
    """
    return {v: k for k, v in CHILDREN_CONTAINERS.items()}[cls]


def children_container_from_str(s):
    """
    returns the registered children container class for the given string
    """
    return CHILDREN_CONTAINERS[s]


def children_container_to_data(cc):
    """
    returns the given children container as data
    """
    return dict(
        children_container_key=children_container_to_str(cc.__class__),
        children_container_data=cc.to_data(),
    )


def children_container_from_data(data):
    """
    convert the given children-container-representation-as-data back into an
    instance of the appropriate children-container class
    """
    cc_key = data["children_container_key"]
    cc_data = data["children_container_data"]
    return children_container_from_str(cc_key).from_data(cc_data)
