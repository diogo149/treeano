import nose.tools as nt
import re
import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import monitor_update_ratio


fX = theano.config.floatX


def test_monitor_update_ratio_node():
    network = tn.WeightDecayNode(
        "decay",
        monitor_update_ratio.MonitorUpdateRatioNode(
            "mur",
            tn.SequentialNode(
                "s",
                [tn.InputNode("i", shape=(None, 3)),
                 tn.LinearMappingNode("linear", output_dim=10),
                 tn.AddBiasNode("bias")])),
        weight_decay=1
    ).network()
    network.build()
    mur_net = network["mur"]
    vws = mur_net.find_vws_in_subtree(tags={"monitor"})
    assert len(vws) == 1
    vw, = vws
    assert re.match(".*_2-norm$", vw.name)
    assert re.match(".*linear.*", vw.name)
    assert not re.match(".*bias.*", vw.name)
