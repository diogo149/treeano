import treeano.nodes as tn
from treeano.sandbox.nodes import unbiased_nesterov_momentum as unm


def test_unbiased_nesterov_momentum_node_serialization():
    tn.check_serialization(
        unm.UnbiasedNesterovMomentumNode("a", tn.IdentityNode("i")))


def test_unbiased_nesterov_momentum_node():
    def unbiased_nag(name, children):
        return tn.SGDNode(name,
                          {"cost": children["cost"],
                           "subtree": unm.UnbiasedNesterovMomentumNode(
                              name + "_momentum",
                              children["subtree"])},
                          learning_rate=0.01)
    tn.test_utils.check_updates_node(unbiased_nag)
