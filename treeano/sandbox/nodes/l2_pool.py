import numpy as np
import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn

fX = theano.config.floatX


@treeano.register_node("l2_pool")
class L2PoolNode(treeano.Wrapper1NodeImpl):

    """
    node that takes the L2 norm of the pooled over region
    """

    hyperparameter_names = ("pool_size",)

    def architecture_children(self):
        nodes = [
            tn.SqrNode(self.name + "_sqr"),
            self.raw_children(),
            # convert mean pool to sum pool by multiplying by pool size
            tn.MultiplyConstantNode(self.name + "_mul"),
            tn.SqrtNode(self.name + "_sqrt"),
        ]
        return [tn.SequentialNode(self.name + "_sequential", nodes)]

    def init_state(self, network):
        super(L2PoolNode, self).init_state(network)
        pool_size = network.find_hyperparameter(["pool_size"])
        network.set_hyperparameter(self.name + "_mul",
                                   "value",
                                   # cast to float, to not trigger
                                   # warn_float64
                                   float(np.prod(pool_size)))


def L2Pool2DNode(name, **kwargs):
    l2_kwargs = {}
    if "pool_size" in kwargs:
        l2_kwargs["pool_size"] = kwargs.pop("pool_size")
    return L2PoolNode(
        name,
        tn.MeanPool2DNode(name + "_pool", **kwargs),
        **l2_kwargs)


def DnnL2PoolNode(name, **kwargs):
    l2_kwargs = {}
    if "pool_size" in kwargs:
        l2_kwargs["pool_size"] = kwargs.pop("pool_size")
    return L2PoolNode(
        name,
        tn.DnnMeanPoolNode(name + "_pool", **kwargs),
        **l2_kwargs)
