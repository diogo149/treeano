from treeano import nodes


def test_update_scale_node_serialization():
    # NOTE: setting shape to be list because of json serialization
    # (serializing a tuple results in a list)
    nodes.check_serialization(nodes.UpdateScaleNode(
        "a", nodes.InputNode("a", shape=[3, 4, 5])))
