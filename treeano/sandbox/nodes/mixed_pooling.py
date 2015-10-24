"""
from
"Generalizing Pooling Functions in Convolutional Neural Networks: Mixed,
Gated, and Tree"
http://arxiv.org/abs/1509.08985

NOTE: currently uses DnnPoolNode to work for 2D and 3D
"""

import toolz
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("mixed_pool")
class MixedPoolNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = (
        tuple([x
               for x in tn.DnnPoolNode.hyperparameter_names
               if x != "mode"])
        + ("learnable",))

    def architecture_children(self):
        mean_seq_node = tn.SequentialNode(
            self.name + "_mean_seq",
            [tn.DnnMeanPoolNode(self.name + "_mean_pool"),
             tn.MultiplyConstantNode(self.name + "_mean_const_mult")]
        )

        max_seq_node = tn.SequentialNode(
            self.name + "_max_seq",
            [tn.DnnMaxPoolNode(self.name + "_max_pool"),
             tn.MultiplyConstantNode(self.name + "_max_const_mult")]
        )

        return [tn.ElementwiseSumNode(self.name + "_sum_mixed",
                                      [max_seq_node, mean_seq_node])]

    def init_state(self, network):
        super(MixedPoolNode, self).init_state(network)
        learnable = network.find_hyperparameter(["learnable"], False)

        # TODO parameterize init alpha
        alpha = 0.5
        if learnable:
            inits = list(toolz.concat(network.find_hyperparameters(
                ["inits"],
                [treeano.inits.ConstantInit(alpha)])))

            alpha = network.create_vw(
                "alpha",
                shape=(),
                is_shared=True,
                tags = {"parameter"},
                inits=inits
            ).variable

        network.set_hyperparameter(self.name + "_max_const_mult",
                                   "value",
                                   alpha)

        network.set_hyperparameter(self.name + "_mean_const_mult",
                                   "value",
                                   1 - alpha)
