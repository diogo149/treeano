import treeano


def to_shared_dict(network):
    network.build()
    vws = network[network.root_node.name].find_vws_in_subtree(is_shared=True)
    name_to_shared = {}
    for vw in vws:
        assert vw.name not in name_to_shared
        name_to_shared[vw.name] = vw.variable
    return name_to_shared


def to_value_dict(network):
    shared_dict = to_shared_dict(network)
    return {k: v.get_value() for k, v in shared_dict.items()}


def load_value_dict(network, value_dict, strict=True):
    shared_dict = to_shared_dict(network)
    value_keys = set(value_dict.keys())
    network_keys = set(shared_dict.keys())
    if strict:
        assert value_keys == network_keys
        keys = value_keys
    else:
        keys = set(value_dict.keys()) & set(shared_dict.keys())
    for k in keys:
        shared = shared_dict[k]
        old_val = shared.get_value()
        new_val = value_dict[k]
        assert old_val.shape == new_val.shape
        shared.set_value(new_val)


def to_preallocated_init(network):
    return treeano.inits.PreallocatedInit(to_shared_dict(network))
