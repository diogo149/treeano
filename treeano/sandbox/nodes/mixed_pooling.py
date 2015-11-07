"""
from
"Generalizing Pooling Functions in Convolutional Neural Networks: Mixed,
Gated, and Tree"
http://arxiv.org/abs/1509.08985
"""

import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

from treeano.sandbox.nodes import batch_fold

fX = theano.config.floatX


@treeano.register_node("mixed_pool")
class MixedPoolNode(treeano.Wrapper0NodeImpl):

    """
    mixed max-average pooling

    NOTE: currently uses DnnPoolNode to work for 2D and 3D
    """

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
            alpha = network.create_vw(
                "alpha",
                shape=(),
                is_shared=True,
                tags = {"parameter"},
                default_inits=[treeano.inits.ConstantInit(alpha)],
            ).variable

        network.set_hyperparameter(self.name + "_max_const_mult",
                                   "value",
                                   alpha)

        network.set_hyperparameter(self.name + "_mean_const_mult",
                                   "value",
                                   1 - alpha)


@treeano.register_node("gated_pool_2d")
class GatedPool2DNode(treeano.Wrapper0NodeImpl):

    """
    gated max-average pooling

    NOTE: not sure how this deals with ignore_border
    """

    hyperparameter_names = tuple([x
                                  for x in tn.Pool2DNode.hyperparameter_names
                                  if x != "mode"])

    def architecture_children(self):
        gate_node = tn.SequentialNode(
            self.name + "_gate_seq",
            [batch_fold.AddAxisNode(self.name + "_add_axis", axis=2),
             batch_fold.FoldUnfoldAxisIntoBatchNode(
                self.name + "_batch_fold",
                 # NOTE: using dnn conv, since pooling is normally strided
                 # and the normal conv is slow with strides
                tn.DnnConv2DWithBiasNode(self.name + "_conv",
                                         num_filters=1),
                axis=1),
             batch_fold.RemoveAxisNode(self.name + "_remove_axis", axis=2),
             tn.SigmoidNode(self.name + "_gate_sigmoid")]
        )

        inverse_gate_node = tn.SequentialNode(
            self.name + "_max_gate",
            [tn.ReferenceNode(self.name + "_gate_ref",
                              reference=gate_node.name),
             tn.MultiplyConstantNode(self.name + "_", value=-1),
             tn.AddConstantNode(self.name + "_add1", value=1)])

        mean_node = tn.ElementwiseProductNode(
            self.name + "_mean_product",
            [tn.MeanPool2DNode(self.name + "_mean_pool"),
             gate_node])

        max_node = tn.ElementwiseProductNode(
            self.name + "_max_product",
            [tn.MaxPool2DNode(self.name + "_max_pool"),
             inverse_gate_node])

        return [tn.ElementwiseSumNode(self.name + "_sum",
                                      [mean_node, max_node])]

    def init_state(self, network):
        super(GatedPool2DNode, self).init_state(network)
        conv_node_name = self.name + "_conv"
        network.forward_hyperparameter(conv_node_name,
                                       "filter_size",
                                       ["pool_size"])
        has_stride = network.maybe_forward_hyperparameter(conv_node_name,
                                                          "conv_stride",
                                                          ["pool_stride",
                                                           "stride"])
        # by default, the stride of a pool is the same as the pool_size
        if not has_stride:
            network.forward_hyperparameter(conv_node_name,
                                           "conv_stride",
                                           ["pool_size"])
        network.maybe_forward_hyperparameter(conv_node_name,
                                             "conv_pad",
                                             ["pool_pad", "pad"])
