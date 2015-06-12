import treeano


def to_shared_dict(network):
    vws = network[network.root_node.name].find_vws_in_subtree(is_shared=True)
    name_to_shared = {}
    for vw in vws:
        assert vw.name not in name_to_shared
        name_to_shared[vw.name] = vw.variable
    return name_to_shared


def to_preallocated_init(network):
    return treeano.inits.PreallocatedInit(to_shared_dict(network))
