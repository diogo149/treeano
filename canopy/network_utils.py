from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import treeano


def to_shared_dict(network):
    network.build()
    if not network.is_relative():
        network = network[network.root_node.name]
    vws = network.find_vws_in_subtree(is_shared=True)
    name_to_shared = {}
    for vw in vws:
        assert vw.name not in name_to_shared
        # if vw.name != vw.variable.name, preallocated init will break
        assert vw.name == vw.variable.name
        name_to_shared[vw.name] = vw.variable
    return name_to_shared


def to_value_dict(network):
    shared_dict = to_shared_dict(network)
    return {k: v.get_value() for k, v in shared_dict.items()}


def load_value_dict(network,
                    value_dict,
                    strict_keys=True,
                    ignore_different_shape=False):
    """
    strict_keys:
    whether or not the network must have the exact same set of keys as the
    value_dict
    """
    shared_dict = to_shared_dict(network)
    value_keys = set(value_dict.keys())
    network_keys = set(shared_dict.keys())
    if strict_keys:
        assert value_keys == network_keys
        keys = value_keys
    else:
        keys = set(value_dict.keys()) & set(shared_dict.keys())

    loaded = 0
    for k in keys:
        shared = shared_dict[k]
        old_val = shared.get_value()
        new_val = value_dict[k]
        if ignore_different_shape:
            if old_val.shape != new_val.shape:
                continue
        else:
            assert old_val.shape == new_val.shape
        shared.set_value(new_val)
        loaded += 1
    print("loaded %d keys (out of %d in value dict, %d in network)"
          % (loaded, len(value_dict), len(shared_dict)))


def to_preallocated_init(network):
    return treeano.inits.PreallocatedInit(to_shared_dict(network))


def num_parameters(network):
    """
    returns the number of "parameter"s in a network
    """
    vws = network.relative_network().find_vws_in_subtree(tags=["parameter"])
    return sum(vw.value.size for vw in vws)
