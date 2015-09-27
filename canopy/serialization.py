import pickle
import os

from . import network_utils


def pickle_network(network, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    root_node = network.root_node
    value_dict = network_utils.to_value_dict(network)
    with open(os.path.join(dirname, "root_node.pkl"), 'wb') as f:
        pickle.dump(root_node, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dirname, "value_dict.pkl"), 'wb') as f:
        pickle.dump(value_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_network(dirname):
    with open(os.path.join(dirname, "root_node.pkl"), 'rb') as f:
        root_node = pickle.load(f)
    with open(os.path.join(dirname, "value_dict.pkl"), 'rb') as f:
        value_dict = pickle.load(f)
    network = root_node.network()
    network_utils.load_value_dict(network, value_dict)
    network.build()
    return network
