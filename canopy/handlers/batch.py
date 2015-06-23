from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import numpy as np
import theano
import theano.tensor as T
import treeano

from . import base


class ChunkVariables(base.NetworkHandlerImpl):

    """
    "chunks" variables for mini-batch evaluation by sending all of the data
    to the GPU at once, but only evaluation one mini-batch at a time

    size of each chunk must be exact divisble by batch_size

    scalar_merge:
    how scalar outputs should be merged together

    cache:
    how to cache inputs for transfer to the GPU
    possible values: "id", "hash"
    use case: datasets that fit in memory can be much more efficient because
    we don't need to send it to the GPU repeatedly
    TODO implement
    """

    def __init__(self,
                 batch_size,
                 variables,
                 scalar_merge="mean",
                 cache="id"):
        # TODO figure out serialization of theano vars
        self.variables = variables
        self.batch_size = batch_size
        if scalar_merge == "mean":
            scalar_merge = np.mean
        elif scalar_merge == "identity":
            scalar_merge = treeano.utils.identity
        self.scalar_merge = scalar_merge
        # TODO use
        self.cache = cache

    def transform_compile_function_kwargs(self, state, **kwargs):
        inputs = kwargs["inputs"]
        givens = kwargs.get("givens")

        if givens is None:
            new_givens = []
        elif isinstance(givens, dict):
            new_givens = list(givens.items())
        elif isinstance(givens, (list, tuple)):
            new_givens = list(givens)

        self.idx_var_ = T.iscalar('batch_idx')
        self.var_to_shared_ = {}
        new_inputs = [self.idx_var_]
        for input_var in inputs:
            if input_var in self.variables:
                v = state.network.network_variable(input_var)
                shared_var = treeano.utils.shared_empty(
                    ndim=v.ndim,
                    dtype=v.dtype,
                )
                idx_slice = slice(self.idx_var_ * self.batch_size,
                                  (self.idx_var_ + 1) * self.batch_size)
                batch_value = shared_var[idx_slice]
                new_givens.append((input_var, batch_value))
                self.var_to_shared_[input_var] = shared_var
            else:
                new_inputs.append(input_var)
        # create a list of variables that are replaced (in order), so that
        # we can later set the value of the chunked shared variables
        # appropriately
        self.shared_list_ = [self.var_to_shared_.get(i, None)
                             for i in inputs]

        kwargs["inputs"] = new_inputs
        kwargs["givens"] = new_givens
        return kwargs

    def call(self, fn, *args, **kwargs):
        assert len(args) == len(self.shared_list_)

        # set shared variables, and keep the non-chunked variables
        new_args = []
        chunk_size = None
        # TODO time transferring data to GPU
        for arg, shared in zip(args, self.shared_list_):
            if shared is None:
                new_args.append(arg)
            else:
                if chunk_size is None:
                    chunk_size = len(arg)
                else:
                    assert len(arg) == chunk_size
                # error if chunk size not a multiple of batch size
                assert (chunk_size % self.batch_size) == 0
                shared.set_value(arg)
        assert chunk_size is not None

        # call function multiple times
        results = []
        for i in range(int(np.ceil(chunk_size / self.batch_size))):
            result = fn(i, *new_args, **kwargs)
            results.append(result)
        res = []
        for outputs in zip(*results):
            if outputs[0].shape:
                res.append(np.concatenate(outputs))
            else:
                res.append(self.scalar_merge(outputs))
        return res

chunk_variables = ChunkVariables
