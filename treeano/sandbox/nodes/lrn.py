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


def local_response_normalization_2d_pool(in_vw, alpha, k, beta, n):
    """
    using built-in pooling
    """
    from theano.tensor.signal.downsample import max_pool_2d
    assert n % 2 == 1, "n must be odd"
    in_var = in_vw.variable
    b, ch, r, c = in_vw.symbolic_shape()
    squared = T.sqr(in_var)
    reshaped = squared.reshape((b, 1, ch, r * c))
    pooled = max_pool_2d(input=reshaped,
                         ds=(n, 1),
                         st=(1, 1),
                         padding=(n // 2, 0),
                         ignore_border=True,
                         mode="average_inc_pad")
    unreshaped = pooled.reshape((b, ch, r, c))
    # multiply by n, since we did a mean pool instead of a sum pool
    return in_var / (((alpha * n) * unreshaped + k) ** beta)


def local_response_normalization_2d_dnn(in_vw, alpha, k, beta, n):
    """
    using cudnn mean pooling
    """
    from theano.sandbox.cuda import dnn
    assert n % 2 == 1, "n must be odd"
    in_var = in_vw.variable
    b, ch, r, c = in_vw.symbolic_shape()
    squared = T.sqr(in_var)
    reshaped = squared.reshape((b, 1, ch, r * c))
    pooled = dnn.dnn_pool(img=reshaped,
                          ws=(n, 1),
                          stride=(1, 1),
                          pad=(n // 2, 0),
                          mode="average_inc_pad")
    unreshaped = pooled.reshape((b, ch, r, c))
    # multiply by n, since we did a mean pool instead of a sum pool
    return in_var / (((alpha * n) * unreshaped + k) ** beta)


def local_response_normalization_pool(in_vw, alpha, k, beta, n):
    """
    using built-in pooling, works for N-D tensors (2D/3D/etc.)
    """
    from theano.tensor.signal.downsample import max_pool_2d
    assert n % 2 == 1, "n must be odd"
    in_var = in_vw.variable
    batch_size, num_channels = in_vw.symbolic_shape()[:2]
    squared = T.sqr(in_var)
    reshaped = squared.reshape((batch_size, 1, num_channels, -1))
    pooled = max_pool_2d(input=reshaped,
                         ds=(n, 1),
                         st=(1, 1),
                         padding=(n // 2, 0),
                         ignore_border=True,
                         mode="average_inc_pad")
    unreshaped = pooled.reshape(in_vw.symbolic_shape())
    # multiply by n, since we did a mean pool instead of a sum pool
    return in_var / (((alpha * n) * unreshaped + k) ** beta)


@treeano.register_node("local_response_normalization_2d")
class LocalResponseNormalization2DNode(treeano.NodeImpl):

    LRN_FUNCTIONS = dict(
        v1=local_response_normalization_2d_v1,
        v2=local_response_normalization_2d_v2,
        pool=local_response_normalization_2d_pool,
        pool_nd=local_response_normalization_pool,
        dnn=local_response_normalization_2d_dnn,
    )

    hyperparameter_names = ("alpha", "k", "beta", "n", "version")

    def compute_output(self, network, in_vw):
        alpha = network.find_hyperparameter(["alpha"], 1e-4)
        k = network.find_hyperparameter(["k"], 2)
        beta = network.find_hyperparameter(["beta"], 0.75)
        n = network.find_hyperparameter(["n"], 5)
        version = network.find_hyperparameter(["version"], "pool")

        lrn_fn = self.LRN_FUNCTIONS[version]

        network.create_vw(
            "default",
            variable=lrn_fn(in_vw,
                            alpha=alpha,
                            k=k,
                            beta=beta,
                            n=n),
            shape=in_vw.shape,
            tags={"output"},
        )


@treeano.register_node("local_response_normalization")
class LocalResponseNormalizationNode(treeano.NodeImpl):

    hyperparameter_names = ("alpha", "k", "beta", "n", "version")

    def compute_output(self, network, in_vw):
        alpha = network.find_hyperparameter(["alpha"], 1e-4)
        k = network.find_hyperparameter(["k"], 2)
        beta = network.find_hyperparameter(["beta"], 0.75)
        n = network.find_hyperparameter(["n"], 5)

        network.create_vw(
            "default",
            variable=local_response_normalization_pool(in_vw,
                                                       alpha=alpha,
                                                       k=k,
                                                       beta=beta,
                                                       n=n),
            shape=in_vw.shape,
            tags={"output"},
        )
