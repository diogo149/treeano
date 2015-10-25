import theano
import theano.tensor as T

from .. import core


@core.register_node("graph")
class GraphNode(core.WrapperNodeImpl):

    """
    creates nodes in an arbitrary graph

    edges:
    - each edge requires at least one of the keys between "from" and "to", and
      can have optional keys "from_key" and "to_key"
    - if "to" is not given, the default is the graph node
    - if "from"'s value is None or not given, the value for this edge is taken
      as the input of the graph node with key "from_key"
    """

    children_container = core.NodesAndEdgesContainer
    hyperparameter_names = ("output_key",)

    def get_input_keys(self, network):
        return [network.find_hyperparameter(["output_key"], "default")]

    def init_state(self, network):
        # create edges
        for original_edge in self._children.edges:
            # one of "to" or "from" must be set
            assert "to" in original_edge or "from" in original_edge
            # add defaults
            edge = {
                "from": None,
                "to": self.name,
                "from_key": "default",
                "to_key": "default",
            }
            edge.update(original_edge)
            assert set(edge.keys()) == {"to", "from", "to_key", "from_key"}
            if edge["from"] is None:
                # None means take the current node's input
                network.forward_input_to(node_name=edge["to"],
                                         previous_to_key=edge["from_key"],
                                         to_key=edge["to_key"],
                                         ignore_no_input=False)
            else:
                network.add_dependency(from_name=edge["from"],
                                       to_name=edge["to"],
                                       from_key=edge["from_key"],
                                       to_key=edge["to_key"])


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

    input_keys = ()

    def init_state(self, network):
        """
        do nothing
        """

    def compute_output(self, network):
        # return variable that always returns an assertion error
        # because the output should not be used
        network.create_vw(
            name="default",
            variable=T.opt.Assert()(T.constant(0.0), 0),
            shape=(),
        )


@core.register_node("auxiliary")
class AuxiliaryNode(core.Wrapper1NodeImpl):

    """
    node that passes its input to an inner node, but returns its input
    """

    # return original input instead of child output
    input_keys = ("default",)
