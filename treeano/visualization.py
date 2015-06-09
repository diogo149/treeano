import pylab
import networkx as nx


def _plot_graph(graph, filename=None, node_size=500):
    nx.draw_networkx(
        graph,
        nx.graphviz_layout(graph),
        node_size=node_size)
    if filename is None:
        pylab.show()
    else:
        pylab.savefig(filename)


def plot_architectural_tree(network, *args, **kwargs):
    return _plot_graph(network.graph.architectural_tree, *args, **kwargs)


def plot_computation_graph(network, *args, **kwargs):
    return _plot_graph(network.graph.computation_graph, *args, **kwargs)
