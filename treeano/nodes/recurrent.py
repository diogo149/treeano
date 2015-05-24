from .. import core
from . import containers
from . import scan


@core.register_node("simple_recurrent")
class SimpleRecurrentNode(core.Wrapper1NodeImpl):

    """
    simple recurrent net with a single hidden layer with a self connection

    takes in a single node to be used as the activation function

    eg.
         v--^
         |  |
    x ---> h

    the input is x as a sequence, the output is h as a sequence
    """

    # FIXME
    hyperparameter_names = ("num_units",)

    def architecture_children(self):
        children = super(SimpleRecurrentNode, self).architecture_children()
        activation_node, = children

        # FIXME
        scan_node = scan.ScanNode(
            self._name + "_scan",
            containers.SequentialNode(
                self._name + "_sequential",
                [containers.SplitCombineNode(
                    self._name + "_splitcombine",
                    [
                        # FIXME
                    ]),
                 activation_node,
                 ]))
        return [scan_node]
