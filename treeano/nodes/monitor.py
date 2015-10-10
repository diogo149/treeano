"""
nodes for creating monitor variables
"""

import theano
import theano.tensor as T

from .. import core
from .. import utils


@core.register_node("monitor_variance")
class MonitorVarianceNode(core.NodeImpl):

    def compute_output(self, network, in_vw):
        super(MonitorVarianceNode, self).compute_output(network, in_vw)
        if network.find_hyperparameter(["monitor"]):
            network.create_vw(
                "var",
                variable=T.var(in_vw.variable),
                shape=(),
                tags={"monitor"},
            )
