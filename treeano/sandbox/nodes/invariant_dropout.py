"""
from
"Making Dropout Invariant to Transformations of Activation Functions and
Inputs"
http://www.dlworkshop.org/56.pdf?attredirects=0
"""
import numpy as np
import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


fX = theano.config.floatX


@treeano.register_node("invariant_dropout")
class InvariantDropoutNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = (tn.DropoutNode.hyperparameter_names
                            + tn.AddBiasNode.hyperparameter_names)

    def architecture_children(self):
        bias_node = tn.AddBiasNode(self.name + "_bias")
        dropout_node = tn.DropoutNode(self.name + "_dropout")
        return [tn.SequentialNode(
            self.name + "_sequential",
            [bias_node,
             dropout_node])]
