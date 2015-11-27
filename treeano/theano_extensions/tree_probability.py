"""
from "Deep Neural Decision Forests"
http://research.microsoft.com/apps/pubs/default.aspx?id=255952
"""

import numpy as np
import theano
import theano.tensor as T


TREE_CACHE = {}


def is_power_of_2(num):
    # bit arithmetic check
    return (num > 0) and ((num & (num - 1)) == 0)


def size_to_tree(size):
    if size in TREE_CACHE:
        return TREE_CACHE[size]

    num_outputs = size + 1
    assert is_power_of_2(num_outputs)

    def inner(lower, upper):
        # lower bound is inclusive, upper bound is exclusive
        # eg. lower = 0, upper = 4
        if lower == upper - 1:
            return []
        midpoint = (lower + upper) // 2
        assert isinstance(midpoint, int)
        assert midpoint * 2 == lower + upper
        return ([(lower, midpoint, upper)]
                + inner(lower, midpoint)
                + inner(midpoint, upper))

    res = inner(0, num_outputs)
    assert len(res) == size
    TREE_CACHE[size] = res
    return res


class TreeProbabilityOp(theano.Op):

    def make_node(self, split_probabilities):
        x = theano.tensor.as_tensor_variable(split_probabilities)
        return theano.gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        left_probabilities, = inputs_storage
        z, = output_storage

        size = left_probabilities.shape[1]
        num_outputs = size + 1
        assert is_power_of_2(num_outputs)

        right_probabilities = 1 - left_probabilities

        tree = size_to_tree(size)

        # allocate result
        output_shape = list(left_probabilities.shape)
        output_shape[1] += 1
        res = np.ones(tuple(output_shape), dtype=left_probabilities.dtype)

        # traverse tree, multiplying probabilities
        for idx, (l, m, r) in enumerate(tree):
            # add new axis so that probabilities broadcast
            res[:, l:m] *= left_probabilities[:, idx, np.newaxis]
            res[:, m:r] *= right_probabilities[:, idx, np.newaxis]

        # save result
        z[0] = res

    def infer_shape(self, node, input_shapes):
        output_shape, = input_shapes
        output_shape = list(output_shape)
        # goes from 2^depth - 1 to 2^depth
        output_shape[1] += 1
        return [tuple(output_shape)]

    def grad(self, inputs, output_grads):
        return [tree_probability_grad(inputs[0],
                                      self(inputs[0]),
                                      output_grads[0])]


class TreeProbabilityGradOp(theano.Op):

    def make_node(self, split_probabilities, outputs, grads):
        x = theano.tensor.as_tensor_variable(split_probabilities)
        return theano.gof.Apply(self, [x, outputs, grads], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        split_probabilities, outputs, grads = inputs_storage
        z, = output_storage

        tree = size_to_tree(split_probabilities.shape[1])

        # NOTE: this part could be done in theano, so that it can be
        # performed on a GPU
        grad_times_output = outputs * grads

        # NOTE: could also be done on GPU
        cumsum = np.cumsum(grad_times_output, axis=1)

        def accumulated_sum(lower, upper):
            # NOTE: lower is inclusive, upper is exclusive
            if lower == 0:
                return cumsum[:, upper - 1]
            else:
                return cumsum[:, upper - 1] - cumsum[:, lower - 1]

        # TODO parameterize
        epsilon = 1e-8

        res = np.zeros_like(split_probabilities)
        # traverse tree, computing gradients
        for idx, (l, m, r) in enumerate(tree):
            p = split_probabilities[:, idx]
            left_grad = accumulated_sum(l, m) / np.clip(p, epsilon, 1)
            right_grad = accumulated_sum(m, r) / np.clip(p - 1, -1, -epsilon)
            res[:, idx] = left_grad + right_grad

        # save result
        z[0] = res

    def infer_shape(self, node, input_shapes):
        # return output same size as split_probabilities
        return [input_shapes[0]]

tree_probability = TreeProbabilityOp()
tree_probability_grad = TreeProbabilityGradOp()
