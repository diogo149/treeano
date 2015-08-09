import treeano
import treeano.nodes as tn
import canopy


def create_big_node_graph(levels):
    assert levels >= 0
    if levels == 0:
        return tn.IdentityNode("i")
    else:
        prev = create_big_node_graph(levels - 1)
        return tn.SequentialNode(
            "s",
            [canopy.node_utils.suffix_node(prev, "0"),
             canopy.node_utils.suffix_node(prev, "1")])


"""
20150808 results:

%timeit create_big_node_graph(5)
# cached_walk = True => 49.7ms
# cached_walk = False => 254ms

%timeit create_big_node_graph(10)
# cached_walk = True => 1.77s
# cached_walk = False => 16.5s
"""
