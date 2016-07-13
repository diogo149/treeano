import numpy as np
import theano
import theano.tensor as T


def group_irregular_length_tensors(tensors):
    """
    groups a set of irregular length tensors (where the length of the first
    axis of each is different) into one large tensor, and also returns
    a vector of the lengths of the original tensors (for use with
    ungroup_irregular_length_tensors)
    """
    shapes = [t.shape for t in tensors]
    # shape on all axes except first must be the same
    for s in shapes:
        assert s[1:] == shapes[0][1:]
    lengths = np.array([s[0] for s in shapes])
    grouped = np.concatenate(tensors, axis=0)
    return grouped, lengths


def ungroup_irregular_length_numpy(x, lengths, pad=True):
    """
    ungroups a grouped irregular length numpy tensor into
    a list of tensors

    pad: if False, returns a list of tensors with different shape.
    if True, returns a single tensor with 0 padding
    """
    assert lengths.ndim == 1
    if pad:
        res_shape = lengths.shape + (lengths.max(),) + x.shape[1:]
        res = np.zeros(res_shape, dtype=x.dtype)
        start_idx = 0
        for idx, l in enumerate(lengths.astype(int)):
            end_idx = start_idx + l
            res[idx, :l] = x[start_idx:end_idx]
            start_idx = end_idx
        return res
    else:
        res = []
        start_idx = 0
        for idx, l in enumerate(lengths.astype(int)):
            end_idx = start_idx + l
            res.append(x[start_idx:end_idx])
            start_idx = end_idx
        return res


class UngroupIrregularLengthTensorsOp(theano.Op):

    """
    undoes group_irregular_length_tensors into a tensor where the second axis
    axis has the length of the largest of the original tensors and pads
    the missing values
    """

    def make_node(self, x, lengths):
        x = theano.tensor.as_tensor_variable(x)
        lengths = theano.tensor.as_tensor_variable(lengths)
        # the first 2 axes of the output should not be broadcastable
        assert not x.broadcastable[0]
        output_broadcastable = (False,) + x.broadcastable
        output_type = T.TensorType(dtype=x.dtype,
                                   broadcastable=output_broadcastable)
        return theano.gof.Apply(self, [x, lengths], [output_type()])

    def perform(self, node, inputs, output_storage):
        x, lengths = inputs
        z, = output_storage
        z[0] = ungroup_irregular_length_numpy(x, lengths, pad=True)

    def grad(self, inputs, output_grads):
        return [ungroup_irregular_length_tensors_grad(inputs[0],
                                                      inputs[1],
                                                      output_grads[0]),
                theano.gradient.grad_undefined(op=self,
                                               x_pos=1,
                                               x=inputs[1])]


class UngroupIrregularLengthTensorsGradOp(theano.Op):

    """
    gradient of UngroupIrregularLengthTensorsOp wrt input tensor
    """

    def make_node(self, x, lengths, output_grad):
        x = theano.tensor.as_tensor_variable(x)
        lengths = theano.tensor.as_tensor_variable(lengths)
        output_grad = theano.tensor.as_tensor_variable(output_grad)
        return theano.gof.Apply(self,
                                [x.shape, lengths, output_grad],
                                [x.type()])

    def perform(self, node, inputs, output_storage):
        x_shape, lengths, output_grad = inputs
        z, = output_storage

        res = np.zeros(x_shape, dtype=output_grad.dtype)

        start_idx = 0
        for idx, l in enumerate(lengths.astype(int)):
            end_idx = start_idx + l
            res[start_idx:end_idx] = output_grad[idx, :l]
            start_idx = end_idx

        z[0] = res


ungroup_irregular_length_tensors = UngroupIrregularLengthTensorsOp()
ungroup_irregular_length_tensors_grad = UngroupIrregularLengthTensorsGradOp()
