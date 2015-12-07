import functools
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


def dropout_max_pool(neibs,
                     axis,
                     dropout_probability,
                     pool_size,
                     deterministic):
    assert neibs.ndim == 2
    assert axis == 1
    keep_probability = 1 - dropout_probability
    if deterministic:
        neibs_sorted = T.sort(neibs, axis=axis)
        # calculate probability of having each element be the maximum
        probs = np.array([keep_probability * dropout_probability ** i
                          for i in reversed(range(np.prod(pool_size)))],
                         dtype=fX)
        return T.dot(neibs_sorted, probs)
    else:
        # FIXME save state in network
        srng = MRG_RandomStreams()
        mask = srng.binomial(neibs.shape,
                             p=keep_probability,
                             dtype=fX)
        return (neibs * mask).max(axis=axis)


@treeano.register_node("dropout_max_pool_2d")
class DropoutMaxPool2DNode(treeano.Wrapper0NodeImpl):
    """
    from "Towards Dropout Training for Convolutional Neural Networks"
    http://arxiv.org/abs/1512.00242

    NOTE: very slow
    """

    hyperparameter_names = (
        tuple([x
               for x in tn.CustomPool2DNode.hyperparameter_names
               if x != "pool_function"])
        + ("deterministic", "dropout_probability", "p"))

    def architecture_children(self):
        return [tn.CustomPool2DNode(self.name + "_pool2d")]

    def init_state(self, network):
        super(DropoutMaxPool2DNode, self).init_state(network)
        pool_size = network.find_hyperparameter(["pool_size"])
        deterministic = network.find_hyperparameter(["deterministic"], False)
        p = network.find_hyperparameter(["dropout_probability", "p"], 0)
        pool_fn = functools.partial(dropout_max_pool,
                                    dropout_probability=p,
                                    pool_size=pool_size,
                                    deterministic=deterministic)
        network.set_hyperparameter(self.name + "_pool2d",
                                   "pool_function",
                                   pool_fn)
