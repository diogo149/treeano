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


def squared_error(preds, target):
    return (preds - target) ** 2


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
