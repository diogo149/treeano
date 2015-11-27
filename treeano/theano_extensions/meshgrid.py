"""
from: https://gist.github.com/eickenberg/f1a0e368961ef6d05b5b
by Michael Eickenberg

TODO fix float64 warnings
"""

import theano
import theano.tensor as T

fX = theano.config.floatX


class _nd_grid(object):

    """Implements the mgrid and ogrid functionality for theano tensor
    variables.

    Parameters
    ==========
        sparse : boolean, optional, default=True
            Specifying False leads to the equivalent of numpy's mgrid
            functionality. Specifying True leads to the equivalent of ogrid.
    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, slices):

        ndim = len(slices)
        ranges = [T.arange(sl.start, sl.stop, sl.step or 1)
                  for sl in slices]
        shapes = [tuple([1] * j + [r.shape[0]] + [1] * (ndim - 1 - j))
                  for j, r in enumerate(ranges)]
        ranges = [r.reshape(shape) for r, shape in zip(ranges, shapes)]
        ones = [T.ones_like(r) for r in ranges]
        if self.sparse:
            grids = ranges
        else:
            grids = []
            for i in range(ndim):
                grid = 1
                for j in range(ndim):
                    if j == i:
                        grid = grid * ranges[j]
                    else:
                        grid = grid * ones[j]
                grids.append(grid)
        return grids


mgrid = _nd_grid()
ogrid = _nd_grid(sparse=True)
