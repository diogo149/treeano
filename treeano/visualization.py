import pylab
import networkx as nx
import theano
import theano.tensor as T


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


def pydotprint_network(network,
                       outfile=None,
                       variables=None,
                       include_updates=True,
                       *args,
                       **kwargs):
    network.build()
    if variables is None:
        vws = network.relative_network(
            network.root_node
        ).find_vws_in_subtree()
        variables = [vw.variable for vw in vws]
        if include_updates:
            variables += [v for _, v in network.update_deltas.to_updates()]
    else:
        # TODO search through update deltas for which ones apply to the
        # given variables
        assert not include_updates, ("include_updates is currently only "
                                     "for showing all variables")
        variables = [network.network_variable(v) for v in variables]

    theano.printing.pydotprint(fct=variables,
                               outfile=outfile,
                               *args,
                               **kwargs)
