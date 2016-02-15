import theano
import theano.tensor as T


def pad(x, padding, fill_value=0):
    """
    applies padding to tensor
    """
    input_shape = x.shape
    output_shape = []
    indices = []

    for dim, pad in enumerate(padding):
        try:
            left_pad, right_pad = pad
        except TypeError:
            left_pad = right_pad = pad
        output_shape.append(left_pad + input_shape[dim] + right_pad)
        indices.append(slice(left_pad, left_pad + input_shape[dim]))

    if fill_value:
        out = T.ones(output_shape) * fill_value
    else:
        out = T.zeros(output_shape)
    return T.set_subtensor(out[tuple(indices)], x)
