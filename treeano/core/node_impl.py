from .update_deltas import UpdateDeltas
from .serialization_state import (children_container_to_data,
                                  children_container_from_data)
from .children_container import (ChildrenContainer,
                                 ChildContainer,
                                 ListChildrenContainer,
                                 NoneChildrenContainer)
from .network import MissingHyperparameter
from .node import NodeAPI


class NodeImpl(NodeAPI):

    """
    a nicer interface on top of the minimal NodeAPI, which sane defaults
    that could be overriden

    usage:
    - you probably do not want to override __init__

    example:
    class TreeaNode(NodeImpl):
        # have hyperparameters named a and b
        # default = no hyperparameters
        hyperparameter_names = ("a", "b")
        # specify a ChildrenContainer (if the node will have children)
        # default = no children
        children_container = ListChildrenContainer
        # define input keys to compute_output
        # default = ("default")
        input_keys = ("default", "something_else")

        def compute_output(self, network, in1, in2):
            # in1 and in2 correspond to input_keys
            assert in1.shape == in2.shape
            network.create_vw(
                name="default",
                variable=in1.variable + in2.variable,
                shape=in1.shape,
                tags={"output"},
            )
    node = TreeaNode("foobar", a=3, b=2)
    """

    # by default, have no hyperparameters
    hyperparameter_names = ()
    # by default, have no children
    children_container = NoneChildrenContainer
    # by default, have a single input_key of "default"
    input_keys = ("default",)

    def __init__(self, name, children=None, **kwargs):
        self._name = name
        self._children = self.children_container(children)
        self.hyperparameters = kwargs
        # some validation
        assert isinstance(self._children, ChildrenContainer)
        assert isinstance(self.input_keys, (list, tuple))
        assert isinstance(self.hyperparameter_names, (list, tuple, set))
        for key in kwargs:
            assert key in self.hyperparameter_names, dict(
                name=self.name,
                key=key,
                msg="Incorrect hyperparameter"
            )

    def __repr__(self):
        param_pairs = [("name", self.name)]
        param_pairs.extend(self.hyperparameters.items())
        param_str = ", ".join("%s=%s" % (k, repr(v)) for k, v in param_pairs)
        root = "%s(%s)" % (self.__class__.__name__, param_str)
        # OPTIMIZE
        children = [repr(child).replace("\n", "\n| ")
                    for child in self._children]
        children_str = "\n| ".join([root] + children)
        return children_str

    @property
    def name(self):
        return self._name

    def _to_architecture_data(self):
        return dict(
            name=self.name,
            children=children_container_to_data(self._children),
            hyperparameters=self.hyperparameters,
        )

    @classmethod
    def _from_architecture_data(cls, data):
        return cls(
            name=data['name'],
            children=children_container_from_data(data["children"]).children,
            **data['hyperparameters']
        )

    def get_hyperparameter(self, network, name):
        """
        default implementation that uses the values in self.hyperparameters
        """
        if name in self.hyperparameters:
            return self.hyperparameters[name]
        else:
            raise MissingHyperparameter

    def raw_children(self):
        """
        convenience method to return children as they were inputted
        """
        return self._children.children

    def architecture_children(self):
        """
        by default, return children in children_container
        """
        return list(iter(self._children))

    def init_long_range_dependencies(self, network):
        """
        by default, do not initialize any long range dependencies
        """

    def init_state(self, network):
        """
        by default, do nothing with the state
        """

    def get_input_keys(self, network):
        """
        by default, use values in self.input_keys
        """
        return self.input_keys

    def compute_output(self, network, *args):
        """
        by default, return first input as output
        """
        network.copy_vw(
            name="default",
            previous_vw=args[0],
            tags={"output"},
        )

    def new_update_deltas(self, network):
        """
        an alternative API for providing update deltas without mutation
        """
        return UpdateDeltas()

    def mutate_update_deltas(self, network, update_deltas):
        """
        default implementation of mutate_update_deltas that uses
        new_update_deltas
        """
        update_deltas += self.new_update_deltas(network)


class WrapperNodeImpl(NodeImpl):

    """
    a nicer interface on top of the minimal NodeAPI, which sane defaults
    that could be overriden - specifically for nodes that wrapper other nodes

    usage:
    - you probably do not want to override __init__
    """
    # by default, children is a list
    children_container = ListChildrenContainer
    # by default, input keys return default input and the value created
    # when calling take_output_from
    input_keys = ("final_child_output",)

    def init_state(self, network):
        """
        by default, forward input to first child, and take output from last
        child
        """
        children = self.architecture_children()
        network.forward_input_to(children[0].name)
        network.take_output_from(children[-1].name,
                                 to_key="final_child_output")


class Wrapper1NodeImpl(WrapperNodeImpl):

    """
    a nicer interface on top of the minimal NodeAPI
    for nodes that wrap a single node

    usage:
    - you probably do not want to override __init__
    """
    # by default, children is a single node
    children_container = ChildContainer


class Wrapper0NodeImpl(WrapperNodeImpl):

    """
    a nicer interface on top of the minimal NodeAPI for nodes that
    don't take in children, but have children (ie. return something from
    self.architecture_children())

    usage:
    - you probably do not want to override __init__
    """
    # by default, children is a single node
    children_container = NoneChildrenContainer
