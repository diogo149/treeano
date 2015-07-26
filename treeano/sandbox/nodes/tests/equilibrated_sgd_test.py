import treeano.nodes as tn
from treeano.sandbox.nodes import equilibrated_sgd


def test_equilibrated_sgd_node_serialization():
    tn.check_serialization(equilibrated_sgd.EquilibratedSGDNode("a"))


def test_equilibrated_sgd_node():
    tn.test_utils.check_updates_node(equilibrated_sgd.EquilibratedSGDNode,
                                     activation="sigmoid")
