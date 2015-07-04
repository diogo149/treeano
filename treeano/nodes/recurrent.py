import theano
import theano.tensor as T

from .. import utils
from .. import core
from . import simple
from . import containers
from . import composite
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

    http://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks
    """

    hyperparameter_names = (("num_units",
                             "batch_size")
                            + scan.ScanNode.hyperparameter_names)

    def architecture_children(self):
        # set activation node as only child
        children = super(SimpleRecurrentNode, self).architecture_children()
        activation_node, = children

        scan_node = scan.ScanNode(
            self._name + "_scan",
            containers.SequentialNode(
                self._name + "_sequential",
                [
                    composite.DenseCombineNode(
                        self._name + "_densecombine",
                        [
                            # mapping from input to hidden state
                            simple.IdentityNode(self._name + "_XforH"),
                            # mapping from previous hidden state to hidden
                            # state
                            containers.SequentialNode(
                                self._name + "_innersequential",
                                [simple.ConstantNode(
                                    self._name + "_initialstate"),
                                 scan.ScanStateNode(
                                     self._name + "_recurrentstate",
                                     # setting new state as output of the
                                     # activation
                                     next_state=activation_node.name),
                                 ])
                        ]
                    ),
                    activation_node,
                ]))
        return [scan_node]

    def init_state(self, network):
        super(SimpleRecurrentNode, self).init_state(network)
        num_units = network.find_hyperparameter(["num_units"])
        # FIXME use batch_axis instead of batch_size
        batch_size = network.find_hyperparameter(["batch_size"])
        if batch_size is None:
            shape = (num_units,)
        else:
            shape = (batch_size, num_units)
        zeros = T.zeros(shape)
        # unfortunately, theano.tensor.zeros makes the result broadcastable
        # if the shape of any dimension is 1, so we have to undo this
        value = T.patternbroadcast(zeros, (False,) * len(shape))
        network.set_hyperparameter(self._name + "_initialstate",
                                   "constant_value",
                                   value)
