import theano.tensor as T
import treeano


def local_response_normalization_2d_v1(in_vw, alpha, k, beta, n):
    """
    cross-channel local response normalization for 2D feature maps
    - input is bc01

    output[i]
    = value of the i-th channel
    = input[i] / (k + alpha * sum(input[j]^2 for j) ** beta)
      - where j is over neighboring channels (from i - n // 2 to i + n // 2)

    This code is adapted from pylearn2.
    https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """
    assert n % 2 == 1, "n must be odd"
    in_var = in_vw.variable
    b, ch, r, c = in_vw.symbolic_shape()
    half_n = n // 2
    input_sqr = T.sqr(in_var)
    extra_channels = T.alloc(0., b, ch + 2 * half_n, r, c)
    input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n + ch, :, :],
                                input_sqr)
    scale = k
    for i in range(n):
        scale += alpha * input_sqr[:, i:i + ch, :, :]
    scale = scale ** beta
    return in_var / scale


def local_response_normalization_2d_v2(in_vw, alpha, k, beta, n):
    """
    cross-channel local response normalization for 2D feature maps
    - input is bc01

    output[i]
    = value of the i-th channel
    = input[i] / (k + alpha * sum(input[j]^2 for j) ** beta)
      - where j is over neighboring channels (from i - n // 2 to i + n // 2)

    This code is adapted from pylearn2.
    https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """
    assert n % 2 == 1, "n must be odd"
    in_var = in_vw.variable
    b, ch, r, c = in_vw.symbolic_shape()
    half_n = n // 2
    input_sqr = T.sqr(in_var)
    extra_channels = T.zeros((b, ch + 2 * half_n, r, c))
    input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n + ch, :, :],
                                input_sqr)
    scale = k + alpha * treeano.utils.smart_sum([input_sqr[:, i:i + ch, :, :]
                                                 for i in range(n)])
    scale = scale ** beta
    return in_var / scale


local_response_normalization_2d = local_response_normalization_2d_v2


@treeano.register_node("local_response_normalization_2d")
class LocalResponseNormalization2DNode(treeano.NodeImpl):

    hyperparameter_names = ("alpha", "k", "beta", "n")

    def compute_output(self, network, in_vw):
        alpha = network.find_hyperparameter(["alpha"], 1e-4)
        k = network.find_hyperparameter(["k"], 2)
        beta = network.find_hyperparameter(["beta"], 0.75)
        n = network.find_hyperparameter(["n"], 5)

        network.create_variable(
            "default",
            variable=local_response_normalization_2d(in_vw,
                                                     alpha=alpha,
                                                     k=k,
                                                     beta=beta,
                                                     n=n),
            shape=in_vw.shape,
            tags={"output"},
        )
