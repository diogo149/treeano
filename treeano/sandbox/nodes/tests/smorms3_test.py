import treeano.nodes as tn
from treeano.sandbox.nodes import smorms3


def test_smorms3_node_serialization():
    tn.check_serialization(smorms3.SMORMS3Node("a"))


def test_smorms3_node():
    tn.test_utils.check_updates_node(smorms3.SMORMS3Node)
