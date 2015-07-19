import numpy as np
import theano
import theano.tensor as T


class PercentileOp(theano.Op):

    """
    like numpy.percentile
    returns q-th percentile of the data for q in [0, 100]
    """

    def make_node(self, a, q):
        if isinstance(q, (int, float)):
            q = theano.gof.Constant(T.fscalar, q)
        return theano.gof.Apply(self, [a, q], [T.fscalar()])

    def perform(self, node, inputs, output_storage):
        a, q = inputs
        z, = output_storage
        z[0] = np.percentile(a, q)


percentile = PercentileOp()
