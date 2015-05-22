from .. import core


@core.register_node("sequential")
class SequentialNode(core.WrapperNodeImpl):

    """
    applies several nodes sequentially
    """

    def init_state(self, network):
        # pass input to first child and get output from last child
        super(SequentialNode, self).init_state(network)
        # set dependencies of children sequentially
        children_names = [c.name for c in self.architecture_children()]
        for from_name, to_name in zip(children_names,
                                      children_names[1:]):
            network.add_dependency(from_name, to_name)


@core.register_node("container")
class ContainerNode(core.WrapperNodeImpl):

    """
    holds several nodes together without explicitly creating dependencies
    between them
    """

    input_keys = ("first_child_output",)

    def init_state(self, network):
        # by default, returns the output of its first child
        # ---
        # this was done because it's a sensible default, and other nodes
        # assume that every node has an output
        # additionally, returning the input of this node didn't work, because
        # sometimes the node has no input (eg. if it contains the input
        # node)
        children = self.architecture_children()
        network.take_output_from(children[0].name,
                                 to_key="first_child_output")
