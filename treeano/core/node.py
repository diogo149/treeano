import abc

import six

from .network import build_network


class NodeAPI(six.with_metaclass(abc.ABCMeta, object)):

    """
    raw API for Node's in the graph
    """

    def __hash__(self):
        return hash((self.__class__, self.name))

    @abc.abstractproperty
    def name(self):
        """
        returns the node's name

        NOTE: should not change over time
        """

    @abc.abstractmethod
    def _to_architecture_data(self):
        """
        returns representation of the architecture data contained in the node
        """

    @abc.abstractmethod
    def _from_architecture_data(cls):
        """
        convert architecture data contained in a node into an instance
        of the appropriate class

        overriding this will allow for reading in old architectures in a
        backwards compatible manner

        NOTE: should be a classmethod
        """

    @abc.abstractmethod
    def get_hyperparameter(self, network, name):
        """
        returns the value of the given hyperparameter, if it is defined for
        the current node, otherwise raises a MissingHyperparameter
        """

    @abc.abstractmethod
    def architecture_children(self):
        """
        returns all child nodes of the given node in the architectural tree
        """

    @abc.abstractmethod
    def init_state(self, network):
        """
        defines all additional state (eg. parameters), possibly in a lazy
        manner, as well as additional dependencies

        also for defining other stateful values (eg. initialization schemes,
        update schemes)

        can assume that all dependencies have completed their init_state call

        NOTE: will be called exactly once
        """

    @abc.abstractmethod
    def get_input_keys(self, network):
        """
        returns the keys of the inputs to compute_output

        NOTE: should not change over time
        """

    @abc.abstractmethod
    def compute_output(self, network, *args):
        """
        computes output of a node as a dictionary from string key to
        output VariableWrapper

        NOTE: will be called exactly once
        """

    @abc.abstractmethod
    def mutate_update_deltas(self, network, update_deltas):
        """
        computes updates of a node and modifies the update_deltas that
        the network passed in

        NOTE: will be called exactly once, and should not mutate the input
        update_deltas, not return a new one
        """

    # ------------------------------------------------------------
    # utilities that are not part of the core API but nice to have
    # ------------------------------------------------------------

    def build(self):
        return build_network(self)
