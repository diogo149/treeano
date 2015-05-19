import abc

import six
import serialization_state


class ChildrenContainer(six.with_metaclass(abc.ABCMeta, object)):

    """
    API for dealing with the children of nodes (which are also nodes)
    """

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def to_data(self):
        """
        returns representation of container as data
        """

    @abc.abstractmethod
    def from_data(cls, data):
        """
        converts data representation back into an instance of the appropriate
        class

        NOTE: should be a classmethod
        """


@serialization_state.register_children_container("list")
class ListChildrenContainer(ChildrenContainer):

    """
    contains children as a list
    """

    def __init__(self, children):
        assert isinstance(children, (list, tuple))
        self.children = children

    def __iter__(self):
        return (x for x in self.children)

    def to_data(self):
        return [serialization_state.node_to_data(child_node) for child_node in self.children]

    @classmethod
    def from_data(cls, data):
        return cls([serialization_state.node_from_data(datum) for datum in data])


@serialization_state.register_children_container("none")
class NoneChildrenContainer(ChildrenContainer):

    """
    contains children as a list
    """

    def __init__(self, children):
        assert children is None

    def __iter__(self):
        return iter([])

    def to_data(self):
        return None

    @classmethod
    def from_data(cls, data):
        return cls(None)


@serialization_state.register_children_container("single_child")
class ChildContainer(ChildrenContainer):

    """
    contains a single child
    """

    def __init__(self, children):
        self.child = children

    def __iter__(self):
        return iter([self.child])

    def to_data(self):
        return serialization_state.node_to_data(self.child)

    @classmethod
    def from_data(cls, data):
        return cls(serialization_state.node_from_data(data))
