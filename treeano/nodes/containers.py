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
        network.create_variable(
            name="default",
            variable=T.opt.Assert()(T.constant(0.0), 0),
            shape=(),
        )


@core.register_node("splitter")
class SplitterNode(core.WrapperNodeImpl):

    """
    passes the input of the node into each of its children
    """

    input_keys = ()

    def init_state(self, network):
        children = self.architecture_children()
        for child in children:
            network.forward_input_to(child.name)

    def compute_output(self, network):
        # return variable that always returns an assertion error
        # because the output should not be used
        network.create_variable(
            name="default",
            variable=T.opt.Assert()(T.constant(0.0), 0),
            shape=(),
        )


@core.register_node("split_combine")
class SplitCombineNode(core.WrapperNodeImpl):

    """
    passes the input of the node into each of its children
    """

    hyperparameter_names = (SplitterNode.hyperparameter_names
                            + simple.FunctionCombineNode.hyperparameter_names)
    input_keys = ("combine_node_output",)

    def architecture_children(self):
        combine_node = simple.FunctionCombineNode(self.name + "_combiner")
        # adding a container so that the input is not passed to it
        combine_container = ContainerNode(self.name + "_combiner_container",
                                          [combine_node])
        children = super(SplitCombineNode, self).architecture_children()
        new_children = []
        for idx, child in enumerate(children):
            new_children.append(
                SequentialNode(
                    "%s_sequential_%d" % (self.name, idx),
                    [child,
                     simple.SendToNode(
                         "%s_send_to_%d" % (self.name, idx),
                         send_to_reference=combine_node.name,
                         to_key="%s_%d" % (self.name, idx),
                     )]))
        return [SplitterNode(
            self.name + "_splitter",
            [combine_container] + new_children,
        )]

    def init_state(self, network):
        # forward input to child (splitter node)
        super(SplitCombineNode, self).init_state(network)
        network.take_output_from(self.name + "_combiner",
                                 to_key="combine_node_output")
