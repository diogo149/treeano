from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import numpy as np
import theano
import theano.tensor as T
import treeano

from . import base


def datamap_batch_merge(datamaps, scalar_merge="mean"):
    """
    concatenates along the batch axis, and merges scalars using a given method
    """
    if scalar_merge == "mean":
        scalar_merge = np.mean
    elif scalar_merge == "identity":
        scalar_merge = treeano.utils.identity
    res = {}
    for key in datamaps[0].keys():  # assumes at least 1 datamap
        outputs = [r[key] for r in datamaps]
        if treeano.utils.is_ndarray(outputs[0]) and outputs[0].shape:
            res[key] = np.concatenate(outputs)
        else:
            res[key] = scalar_merge(outputs)
    return res


class SplitInput(base.NetworkHandlerImpl):

    """
    Splits the input into multiple smaller inputs along axis 0,
    applies the inner function, and concatenates the results.

    Size of input must be a mulitple of split_size.

    scalar_merge:
    how scalar outputs should be merged together

    """

    def __init__(self,
                 split_size,
                 keys,
                 scalar_merge="mean"):
        self.split_size = split_size
        self.keys = keys
        self.scalar_merge = scalar_merge

    def call(self, fn, in_dict, *args, **kwargs):
        input_size = None
        for input_key in self.keys:
            input_val = in_dict[input_key]
            if input_size is None:
                input_size = len(input_val)
            else:
                assert len(input_val) == input_size
        assert input_size is not None
        results = []
        # optimization to prevent copying and simply pass computation through
        # when splitting is a no-op
        if input_size == self.split_size:
            return fn(in_dict, *args, **kwargs)
        for i in range(int(np.ceil(input_size / self.split_size))):
            inner_map = dict(in_dict)
            split_slice = slice(i * self.split_size, (i + 1) * self.split_size)
            for key in self.keys:
                inner_map[key] = in_dict[key][split_slice]
            result = fn(inner_map, *args, **kwargs)
            results.append(result)
        return datamap_batch_merge(results, scalar_merge=self.scalar_merge)

split_input = SplitInput


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

    BATCH_IDX_KEY = "batch_idx"

    def __init__(self,
                 batch_size,
                 variables,
                 scalar_merge="mean",
                 cache="id",
                 strict_size=True):
        # TODO figure out serialization of theano vars
        self.variables = variables
        self.batch_size = batch_size
        self.scalar_merge = scalar_merge
        # TODO actually use cache
        self.cache = cache
        self.strict_size = strict_size

    def transform_compile_function_kwargs(self, state, **kwargs):
        inputs = kwargs["inputs"]
        givens = kwargs.get("givens")

        assert isinstance(inputs, dict)
        assert self.BATCH_IDX_KEY not in inputs

        if givens is None:
            new_givens = []
        elif isinstance(givens, dict):
            new_givens = list(givens.items())
        elif isinstance(givens, (list, tuple)):
            new_givens = list(givens)

        self.idx_var_ = T.iscalar('batch_idx')
        self.key_to_shared_ = {}
        new_inputs = dict(inputs)
        new_inputs[self.BATCH_IDX_KEY] = self.idx_var_
        for input_key, input_var in inputs.items():
            if input_var in self.variables:
                # remove key from inputs
                new_inputs.pop(input_key)
                # create shared variable
                v = state.network.network_variable(input_var)
                shared_var = treeano.utils.shared_empty(
                    ndim=v.ndim,
                    dtype=v.dtype,
                )
                # create givens for variable
                idx_slice = slice(self.idx_var_ * self.batch_size,
                                  (self.idx_var_ + 1) * self.batch_size)
                batch_value = shared_var[idx_slice]
                new_givens.append((input_var, batch_value))
                # store shared variable
                self.key_to_shared_[input_key] = shared_var

        kwargs["inputs"] = new_inputs
        kwargs["givens"] = new_givens
        return kwargs

    def __call__(self, state, in_dict, *args, **kwargs):
        # set shared variables, and keep the non-chunked variables
        chunk_size = None
        # make a copy, since we are mutating it
        in_dict = dict(in_dict)
        with state.time("data_transfer"):
            for input_key, shared in self.key_to_shared_.items():
                input_val = in_dict.pop(input_key)
                if chunk_size is None:
                    chunk_size = len(input_val)
                else:
                    assert len(input_val) == chunk_size
                if self.strict_size:
                    # error if chunk size not a multiple of batch size
                    assert (chunk_size % self.batch_size) == 0
                shared.set_value(input_val)
        assert chunk_size is not None

        # call function multiple times
        # TODO maybe factor this out
        results = []
        for i in range(int(np.ceil(chunk_size / self.batch_size))):
            in_dict[self.BATCH_IDX_KEY] = i
            result = self._inner_handler(state, in_dict, *args, **kwargs)
            results.append(result)
        res = datamap_batch_merge(results, scalar_merge=self.scalar_merge)
        # free memory
        # may be inefficient if not necessary
        with state.time("data_free"):
            for shared in self.key_to_shared_.values():
                shared.set_value(np.zeros([0] * shared.ndim,
                                          dtype=shared.dtype))
        return res

chunk_variables = ChunkVariables


class BatchPad(base.NetworkHandlerImpl):

    """
    pads variables with 0's to the specified batch size
    """

    def __init__(self,
                 batch_size,
                 keys,
                 axis=0):
        # TODO figure out serialization of theano vars
        self.keys = keys
        self.batch_size = batch_size
        self.axis = axis

    def _pad(self, arr):
        rem = arr.shape[self.axis] % self.batch_size
        if rem == 0:
            return arr
        else:
            pad_shape = [s if i != self.axis else (self.batch_size - rem)
                         for i, s in enumerate(arr.shape)]
            to_pad = np.zeros(pad_shape, dtype=arr.dtype)
            return np.concatenate([arr, to_pad], axis=self.axis)

    def call(self, fn, in_dict, *args, **kwargs):
        # make a copy, since we are mutating it
        in_dict = dict(in_dict)
        for key in self.keys:
            in_dict[key] = self._pad(in_dict[key])

        return fn(in_dict, *args, **kwargs)

batch_pad = BatchPad
