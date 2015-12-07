import theano
import theano.tensor as T

import treeano
import treeano.nodes as tn
import treeano.sandbox.utils


@treeano.register_node("fold_axis_into_batch")
class FoldAxisIntoBatchNode(treeano.NodeImpl):

    """
    folds a given axis into a batch, to allow batch processing across that
    axis in a shared manner
    """
    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        assert 0 < axis < in_vw.ndim
        var = in_vw.variable

        if axis != 1:
            # move axis to first non-batch axis
            pattern = list(range(in_vw.ndim))
            pattern.remove(axis)
            pattern.insert(1, axis)
            var = var.dimshuffle(*pattern)

        # reshape axis into batch
        new_ss = list(in_vw.symbolic_shape())
        new_ss[0] *= new_ss.pop(axis)
        out_var = var.reshape(tuple(new_ss))

        # calculate new shape
        new_s = list(in_vw.shape)
        axis_length = new_s.pop(axis)
        if axis_length is None or new_s[0] is None:
            new_s[0] = None
        else:
            new_s[0] *= axis_length

        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(new_s),
            tags={"output"}
        )


@treeano.register_node("fold_unfold_axis_into_batch")
class FoldUnfoldAxisIntoBatchNode(treeano.Wrapper1NodeImpl):

    """
    temporarily folds a given axis into a batch

    use cases:
    - applying function across recurrent dim
    - applying 2D convnet to 3D data
    """

    hyperparameter_names = ("axis",)
    input_keys = ("default",) + treeano.Wrapper1NodeImpl.input_keys

    def architecture_children(self):
        node = self.raw_children()
        assert isinstance(node, treeano.core.NodeAPI)
        return [tn.SequentialNode(
            self.name + "_sequential",
            [FoldAxisIntoBatchNode(self.name + "_batchfold"),
             node])]

    def compute_output(self, network, original_vw, in_vw):
        axis = network.find_hyperparameter(["axis"])
        assert 0 < axis < min(in_vw.ndim + 1, original_vw.ndim)
        var = in_vw.variable

        # reshape axis out of batch
        original_ss = original_vw.symbolic_shape()
        in_ss = in_vw.symbolic_shape()
        var = var.reshape((original_ss[0], original_ss[axis]) + in_ss[1:])

        # TODO parameterize final axis, in case we want this to be different
        # from original axis
        if axis != 1:
            # move axis back to appropriate location
            pattern = list(range(in_vw.ndim + 1))
            pattern.pop(1)
            pattern.insert(axis, 1)
            var = var.dimshuffle(*pattern)

        # calculate new shape
        original_shape = original_vw.shape
        new_shape = list(in_vw.shape)
        new_shape[0] = original_shape[0]
        new_shape.insert(axis, original_shape[axis])
        new_shape = tuple(new_shape)

        network.create_vw(
            "default",
            variable=var,
            shape=new_shape,
            tags={"output"},
        )


@treeano.register_node("add_axis")
class AddAxisNode(treeano.NodeImpl):

    """
    adds an axis with length 1
    """
    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        assert 0 <= axis <= in_vw.ndim

        # calculate variable
        pattern = list(range(in_vw.ndim))
        pattern.insert(axis, "x")
        out_var = in_vw.variable.dimshuffle(*pattern)

        # calculate shape
        out_shape = list(in_vw.shape)
        out_shape.insert(axis, 1)
        out_shape = tuple(out_shape)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@treeano.register_node("remove_axis")
class RemoveAxisNode(treeano.NodeImpl):

    """
    removes an axis with length 1
    """
    hyperparameter_names = ("axis",)

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        assert 0 <= axis < in_vw.ndim

        # calculate shape
        out_shape = list(in_vw.shape)
        axis_length = out_shape.pop(axis)
        assert axis_length == 1
        out_shape = tuple(out_shape)

        # calculate variable
        ss = list(in_vw.symbolic_shape())
        ss.pop(axis)
        out_var = in_vw.variable.reshape(ss)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )


@treeano.register_node("split_axis")
class SplitAxisNode(treeano.NodeImpl):

    """
    splits a single axis into multiple axes with the given shape
    """
    hyperparameter_names = ("axis", "shape")

    def compute_output(self, network, in_vw):
        axis = network.find_hyperparameter(["axis"])
        shape = network.find_hyperparameter(["shape"])
        assert 0 <= axis < in_vw.ndim

        # calculate shape
        in_shape = in_vw.shape
        shape_with_none = tuple([None if s == -1 else s for s in shape])
        out_shape = in_shape[:axis] + shape_with_none + in_shape[axis + 1:]

        # calculate variable
        in_ss = in_vw.symbolic_shape()
        ss = in_ss[:axis] + shape + in_ss[axis + 1:]
        out_var = in_vw.variable.reshape(ss)

        network.create_vw(
            "default",
            variable=out_var,
            shape=out_shape,
            tags={"output"},
        )
