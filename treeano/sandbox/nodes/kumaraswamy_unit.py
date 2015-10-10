"""
from
"Improving neural networks with bunches of neurons modeled by Kumaraswamy
units: Preliminary study"
http://arxiv.org/abs/1505.02581
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


def kumaraswamy_unit(x, a=8, b=30):
    return 1 - (1 - T.nnet.sigmoid(x) ** a) ** b


@treeano.register_node("kumaraswamy_unit")
class KumaraswamyUnitNode(treeano.NodeImpl):

    hyperparameter_names = ("kumaraswamy_a",
                            "kumaraswamy_b")

    def compute_output(self, network, in_vw):
        a = network.find_hyperparameter(["kumaraswamy_a"], 8)
        b = network.find_hyperparameter(["kumaraswamy_b"], 30)
        network.create_vw(
            "default",
            variable=kumaraswamy_unit(in_vw.variable, a, b),
            shape=in_vw.shape,
            tags={"output"},
        )
