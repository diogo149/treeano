"""
from
"Stochastic Pooling for Regularization of Deep Convolutional Neural Networks"
http://arxiv.org/abs/1301.3557

NOTE: very slow
"""

import functools
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def stochastic_pool(neibs, axis, deterministic):
    """
    NOTE: assumes that inputs are >= 0
    """
    assert axis == 1
    # TODO parameterize
    epsilon = 1e-6
    as_p = neibs / (neibs.sum(axis=axis, keepdims=True) + epsilon)
    if deterministic:
        mask = as_p
    else:
        # FIXME save state in network
        srng = MRG_RandomStreams()
        mask = srng.multinomial(pvals=as_p).astype(fX)
    return (neibs * mask).sum(axis=axis)


@treeano.register_node("stochastic_pool_2d")
class StochasticPool2DNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = (
        tuple([x
               for x in tn.CustomPool2DNode.hyperparameter_names
               if x != "pool_function"])
        + ("deterministic",))

    def architecture_children(self):
        return [tn.CustomPool2DNode(self.name + "_pool2d")]

    def init_state(self, network):
        super(StochasticPool2DNode, self).init_state(network)
        deterministic = network.find_hyperparameter(["deterministic"])
        pool_fn = functools.partial(stochastic_pool,
                                    deterministic=deterministic)
        network.set_hyperparameter(self.name + "_pool2d",
                                   "pool_function",
                                   pool_fn)
