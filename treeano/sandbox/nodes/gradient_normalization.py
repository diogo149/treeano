import theano
import theano.tensor as T
from theano.compile import ViewOp
import treeano
import treeano.nodes as tn
import canopy


class GradientBatchNormalizationOp(ViewOp):

    def __init__(self,
                 normalization_axes=(0,),
                 subtract_mean=False,
                 keep_mean=False,
                 epsilon=1e-8):
        assert isinstance(normalization_axes, (list, tuple))
        self.normalization_axes_ = tuple(normalization_axes)
        self.subtract_mean_ = subtract_mean
        self.keep_mean_ = keep_mean
        self.epsilon_ = epsilon

    def grad(self, inputs, output_gradients):
        old_grad, = output_gradients

        # calculate mean and std
        kwargs = dict(axis=self.normalization_axes_, keepdims=True)
        mean = old_grad.mean(**kwargs)
        std = old_grad.std(**kwargs) + self.epsilon_

        # initialize to old gradient
        new_grad = old_grad
        if self.subtract_mean_:
            new_grad -= mean
        # divide by std
        new_grad /= std
        if self.keep_mean_ and self.subtract_mean_:
            new_grad += mean
        return (new_grad,)


@treeano.register_node("gradient_batch_normalization")
class GradientBatchNormalizationNode(treeano.NodeImpl):

    """
    like treeano.theano_extensions.gradient.gradient_reversal
    """

    hyperparameter_names = ("subtract_mean",
                            "keep_mean",
                            "epsilon")

    def compute_output(self, network, in_vw):
        subtract_mean = network.find_hyperparameter(["subtract_mean"], False)
        keep_mean = network.find_hyperparameter(["keep_mean"], False)
        epsilon = network.find_hyperparameter(["epsilon"], 1e-8)
        # TODO parameterize normalization axes
        normalization_axes = [axis for axis in range(in_vw.ndim) if axis != 1]
        out_var = GradientBatchNormalizationOp(
            normalization_axes=normalization_axes,
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
