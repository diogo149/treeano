import theano
import theano.tensor as T
import treeano
import treeano.nodes as tn


@treeano.register_node("timesout")
class TimesoutNode(treeano.Wrapper0NodeImpl):

    hyperparameter_names = tuple(
        [n for n in tn.FeaturePoolNode.hyperparameter_names
         if n != "pool_function"])

    def architecture_children(self):
        return [tn.FeaturePoolNode(self.name + "_featurepool")]

    def init_state(self, network):
        super(TimesoutNode, self).init_state(network)

        network.set_hyperparameter(self.name + "_featurepool",
                                   "pool_function",
                                   T.prod)


@treeano.register_node("indexed_timesout")
class IndexedTimesoutNode(treeano.NodeImpl):

    def compute_output(self, network, in_vw):
        in_var = in_vw.variable
        half = in_vw.shape[1] // 2
        out_var = in_var[:, :half] * in_var[:, half:]
        out_shape = list(in_vw.shape)
        out_shape[1] = half
        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )


@treeano.register_node("indexed_gated_timesout")
class IndexedGatedTimesoutNode(treeano.NodeImpl):
    """
    similar to forget gate
    """

    def compute_output(self, network, in_vw):
        in_var = in_vw.variable
        half = in_vw.shape[1] // 2
        out_var = in_var[:, :half] * T.nnet.sigmoid(in_var[:, half:])
        out_shape = list(in_vw.shape)
        out_shape[1] = half
        network.create_vw(
            "default",
            variable=out_var,
            shape=tuple(out_shape),
            tags={"output"},
        )


@treeano.register_node("double_timesout")
class DoubleTimesoutNode(tn.BaseActivationNode):

    def activation(self, network, in_vw):
        in_var = in_vw.variable
        return in_var * T.roll(in_var, shift=1, axis=1)


def GeometricMeanOutNode(name, epsilon=1e-8, **kwargs):
    return tn.SequentialNode(
        name,
        [tn.ReLUNode(name + "_relu"),
         tn.AddConstantNode(name + "_add", value=epsilon),
         TimesoutNode(name + "_to", **kwargs),
         tn.SqrtNode(name + "_sqrt"),
         tn.AddConstantNode(name + "_sub", value=-(epsilon ** 2))]
    )


def BiasedTimesoutNode(name, bias=1, **kwargs):
    return tn.SequentialNode(
        name,
        [tn.AddConstantNode(name + "_add", value=bias),
         TimesoutNode(name + "_to", **kwargs),
         tn.AddConstantNode(name + "_sub", value=-(bias ** 2))]
    )


def BiasedDoubleTimesoutNode(name, bias=1, **kwargs):
    return tn.SequentialNode(
        name,
        [tn.AddConstantNode(name + "_add", value=bias),
         DoubleTimesoutNode(name + "_to", **kwargs),
         tn.AddConstantNode(name + "_sub", value=-(bias ** 2))]
    )
