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
        node = self._children.children
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
        in_shape = in_vw.shape
        new_shape = (original_shape[0], original_shape[axis]) + in_shape[1:]

        network.create_vw(
            "default",
            variable=var,
            shape=new_shape,
            tags={"output"},
        )
