import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

fX = theano.config.floatX

# create shared random state to access
srng = RandomStreams()


def all_equal(seq):
    """
    whether or not all elements of a sequence are equal
    """
    return len(set(seq)) == 1


def identity(x):
    return x


def first(f, *args):
    return f


def stable_softmax(x):
    """
    numerical stabilization to avoid f32 overflow
    http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
    """
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out


def squared_error(pred, target):
    return (pred - target) ** 2


def categorical_crossentropy_i32(pred, target):
    """
    like theano.tensor.nnet.categorical_crossentropy, but with clearer
    assertions on the expected input
    """
    assert target.dtype == "int32"
    assert target.ndim == 1
    assert pred.dtype == fX
    assert pred.ndim == 2
    return T.nnet.categorical_crossentropy(pred, target)


def deep_clone(output, replace, **kwargs):
    """
    like theano.clone, but makes sure to replace in the default_update of
    shared variables as well
    """
    new_output = list(output)
    default_update_idxs = []
    for idx, v in enumerate(theano.gof.graph.inputs(output)):
        if hasattr(v, "default_update"):
            new_output.append(v.default_update)
            default_update_idxs.append(idx)
    cloned = theano.clone(new_output, replace, **kwargs)
    cloned_output = cloned[:len(output)]
    cloned_default_updates = cloned[len(output):]
    assert len(cloned_default_updates) == len(default_update_idxs)
    cloned_inputs = theano.gof.graph.inputs(cloned_output)
    for idx, update in zip(default_update_idxs, cloned_default_updates):
        v = cloned_inputs[idx]
        assert hasattr(v, "default_update")
        v.default_update = update
    return cloned_output
