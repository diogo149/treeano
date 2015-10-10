import theano
import theano.tensor as T

from .. import core
from . import simple


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
