import theano
import theano.tensor as T
from theano.compile import ViewOp
import treeano
import treeano.nodes as tn
import canopy


class GradientBatchNormalizationOp(ViewOp):

    def __init__(self,
                 normalization_axes=(0,),
                 mean_preprocess_axes=(),
                 subtract_mean=False,
                 keep_mean=False,
                 epsilon=1e-8):
        assert isinstance(normalization_axes, (list, tuple))
        assert isinstance(mean_preprocess_axes, (list, tuple))
        assert len(set(normalization_axes) & set(mean_preprocess_axes)) == 0
        self.normalization_axes_ = tuple(normalization_axes)
        self.mean_preprocess_axes_ = tuple(mean_preprocess_axes)
        self.subtract_mean_ = subtract_mean
        self.keep_mean_ = keep_mean
        self.epsilon_ = epsilon

    def grad(self, inputs, output_gradients):
        old_grad, = output_gradients
        # initialize to old gradient
        new_grad = old_grad

        # mean preprocess
        if self.mean_preprocess_axes_:
            old_grad = old_grad.mean(axis=self.mean_preprocess_axes_,
                                     keepdims=True)

        # calculate mean and std
        kwargs = dict(axis=self.normalization_axes_, keepdims=True)
        mean = old_grad.mean(**kwargs)
        std = old_grad.std(**kwargs) + self.epsilon_

        # remove mean
        if self.subtract_mean_:
            new_grad -= mean
        # divide by std
        new_grad /= std
        # optionally keep mean
        if self.keep_mean_ and self.subtract_mean_:
            new_grad += mean

        return (new_grad,)


@treeano.register_node("gradient_batch_normalization")
class GradientBatchNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("gbn_normalization_axes",
                            "normalization_axes",
                            "mean_preprocess_axes",
                            "subtract_mean",
                            "keep_mean",
                            "epsilon")

    def compute_output(self, network, in_vw):
        subtract_mean = network.find_hyperparameter(["subtract_mean"], False)
        keep_mean = network.find_hyperparameter(["keep_mean"], False)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        # by default, normalize all except axis 1
        default_normalization_axes = [axis for axis in range(in_vw.ndim)
                                      if axis != 1]
        normalization_axes = network.find_hyperparameter(
            ["gbn_normalization_axes",
             "normalization_axes"],
            default_normalization_axes)
        # TODO experiment if meaning all non-batch axes is useful
        default_mean_preprocess_axes = ()
        mean_preprocess_axes = network.find_hyperparameter(
            ["mean_preprocess_axes"],
            default_mean_preprocess_axes)
        out_var = GradientBatchNormalizationOp(
            normalization_axes=normalization_axes,
            mean_preprocess_axes=mean_preprocess_axes,
            subtract_mean=subtract_mean,
            keep_mean=keep_mean,
            epsilon=epsilon,
        )(in_vw.variable)
        network.create_vw(
            "default",
            variable=out_var,
            shape=in_vw.shape,
            tags={"output"}
        )


def remove_gradient_batch_normalization_handler():
    """
    gradient batch normalization can be slower at test time, where it isn't
    generally used (since updates are not occuring), so this handler
    can remove the nodes

    TODO: figure out why
    """
    return canopy.handlers.remove_nodes_with_class(
        GradientBatchNormalizationNode)
