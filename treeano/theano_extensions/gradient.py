import theano
import theano.tensor as T
from theano.compile import ViewOp


class GradientReversalOp(ViewOp):

    def grad(self, inputs, output_gradients):
        return [-output_gradients[0]]

gradient_reversal = GradientReversalOp()
