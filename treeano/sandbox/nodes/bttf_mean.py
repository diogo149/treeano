# TODO: see if op can change default_updates

import warnings
import theano
import theano.tensor as T
import treeano
from treeano.sandbox.utils import OverwriteGrad


def _backprop_to_the_future_mean_forward(batch_mean,
                                         rolling_mean,
                                         rolling_grad,
                                         alpha):
    return rolling_mean + 0 * (batch_mean + rolling_grad) + 0 * alpha


class BackpropToTheFutureMeanOp(OverwriteGrad):

    def __init__(self, update_averages):
        super(BackpropToTheFutureMeanOp, self).__init__(
            fn=_backprop_to_the_future_mean_forward)
        self.update_averages = update_averages

    def grad(self, inputs, out_grads):
        batch_mean, rolling_mean, rolling_grad, alpha = inputs
        out_grad, = out_grads

        if self.update_averages:
            assert treeano.utils.is_shared_variable(rolling_mean)
            assert treeano.utils.is_shared_variable(rolling_grad)
            # HACK this is super hacky and won't work for certain
            # computation graphs
            # TODO make assertion again
            if (hasattr(rolling_mean, "default_update") or
                    hasattr(rolling_grad, "default_update")):
                warnings.warn("rolling mean/grad already has updates - "
                              "overwritting. this can be caused by calculating "
                              "the gradient of backprop to the future mean "
                              "multiple times")

            rolling_mean.default_update = (alpha * rolling_mean +
                                           (1 - alpha) * batch_mean)
            rolling_grad.default_update = (alpha * rolling_grad +
                                           (1 - alpha) * out_grad)
        else:
            # HACK remove default_update
            if hasattr(rolling_mean, "default_update"):
                delattr(rolling_mean, "default_update")
            if hasattr(rolling_grad, "default_update"):
                delattr(rolling_grad, "default_update")

        return [rolling_grad,
                T.zeros_like(rolling_mean),
                T.zeros_like(rolling_grad),
                T.zeros_like(alpha)]

backprop_to_the_future_mean_with_updates = BackpropToTheFutureMeanOp(
    update_averages=True)
backprop_to_the_future_mean_no_updates = BackpropToTheFutureMeanOp(
    update_averages=False)
